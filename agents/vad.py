# copied from https://github.com/livekit/agents/blob/main/livekit-plugins/livekit-plugins-silero/livekit/plugins/silero/vad.py
# debug cpu usage, start / end events, and return resampled frames

# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import logging
import asyncio
import time
from typing import List, Optional, Callable
from livekit import rtc, agents
import torch
import numpy as np
from collections import deque
from common.config import config


class VAD(agents.vad.VAD):
    def __init__(
        self, *, model_path: Optional[str] = None, use_onnx: bool = True 
    ) -> None:
        if model_path:
            model = torch.jit.load(model_path)
            model.eval()
        else:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                onnx=use_onnx,
                force_reload=False,
                skip_validation=True,
            )
        self._model = model


    def stream(
        self,
        *,
        min_speaking_duration: float = 0.5,
        min_silence_duration: float = 0.5,
        padding_duration: float = 0.1,
        sample_rate: int = 16000,
        max_buffered_speech: float = 45.0,
        threshold: float = 0.5,
        newEventCallback: Optional[Callable[[agents.vad.VADEventType], None]] = None,
    ) -> "VADStream":
        return VADStream(
            self._model,
            min_speaking_duration=min_speaking_duration,
            min_silence_duration=min_silence_duration,
            padding_duration=padding_duration,
            sample_rate=sample_rate,
            max_buffered_speech=max_buffered_speech,
            threshold=threshold,
            newEventCallback=newEventCallback,
        )


# Based on https://github.com/snakers4/silero-vad/blob/94504ece54c8caeebb808410b08ae55ee82dba82/utils_vad.py#L428
class VADStream(agents.vad.VADStream):
    def __init__(
        self,
        model,
        *,
        min_speaking_duration: float,
        min_silence_duration: float,
        padding_duration: float,
        sample_rate: int,
        max_buffered_speech: float,
        threshold: float,
        newEventCallback: Optional[Callable[[agents.vad.VADEvent], None]] = None,
    ) -> None:
        self._min_speaking_duration = min_speaking_duration
        self._min_silence_duration = min_silence_duration
        self._padding_duration = padding_duration
        self._sample_rate = sample_rate
        self._max_buffered_speech = max_buffered_speech
        self._threshold = threshold
        self._newEventCallback: Optional[Callable[[agents.vad.VADEventType], None]] = newEventCallback

        if sample_rate not in [8000, 16000]:
            raise ValueError("Silero VAD only supports 8KHz and 16KHz sample rates")

        self._queue = asyncio.Queue[rtc.AudioFrame]()
        self._event_queue = asyncio.Queue[agents.vad.VADEvent]()
        self._model = model

        self._closed = False
        self._speaking = False
        self._waiting_start = False
        self._waiting_end = False
        self._current_sample = 0
        self._min_speaking_samples = min_speaking_duration * sample_rate
        self._min_silence_samples = min_silence_duration * sample_rate
        self._padding_duration_samples = padding_duration * sample_rate
        self._max_buffered_samples = max_buffered_speech * sample_rate

        self._queued_frames: deque[rtc.AudioFrame] = deque()
        self._original_frames: deque[rtc.AudioFrame] = deque()
        self._buffered_frames: List[rtc.AudioFrame] = []
        self._main_task = asyncio.create_task(self._run())
        self._inference_frame_count = 4
        self._last_event_type = None
        self._speech_probs: List[float] = []
        self._asume_speech_min_prob = config.agents.vad_asume_speech_min_prob
        self._asume_speech_max_count = config.agents.vad_asume_speech_max_count

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                logging.error(f"silero vad task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)

    def statusChanged(self, event_type: agents.vad.VADEventType) -> None:
        if self._last_event_type == event_type:
            return
        self._last_event_type = event_type
        if self._newEventCallback:
            #  self._newEventCallback(event_type)
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._newEventCallback(event_type))
                # loop.call_soon_threadsafe(self._newEventCallback, event_type)
            else:
                loop.run_until_complete(self._newEventCallback(event_type))

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._queue.put_nowait(frame)

    async def flush(self) -> None:
        await self._queue.join()

    async def aclose(self) -> None:
        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task

    async def _run(self):
        while True:
            try:
                frame = await self._queue.get()
            except asyncio.CancelledError:
                break

            self._queue.task_done()

            # resample to silero's sample rate
            if frame.sample_rate != self._sample_rate:
                resampled_frame = frame.remix_and_resample(
                    self._sample_rate, 1
                )  # TODO: This is technically wrong, fix when we have a better resampler
            else:
                resampled_frame = frame
            self._original_frames.append(frame)
            self._queued_frames.append(resampled_frame)

            # run inference by chunks of 40ms until we run out of data
            while True:
                available_length = sum(
                    f.samples_per_channel for f in self._queued_frames
                )

                samples_40ms = self._sample_rate // 1000 * 10 * self._inference_frame_count
                if available_length < samples_40ms:
                    break

                await asyncio.shield(self._run_inference())

        self._closed = True

    async def _run_inference(self) -> None:
        # merge the first 4 frames (we know each is 10ms)
        if len(self._queued_frames) < self._inference_frame_count:
            return

        original_frames = [self._original_frames.popleft() for _ in range(self._inference_frame_count)]
        resampled_frames = [self._queued_frames.popleft() for _ in range(self._inference_frame_count)]
        merged_frame = agents.utils.merge_frames(resampled_frames)

        # convert data_40ms to tensor & f32
        tensor = torch.from_numpy(np.frombuffer(merged_frame.data, dtype=np.int16))
        tensor = tensor.to(torch.float32) / 32768.0

        # run inference
        now = time.time()
        speech_prob = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._model(tensor, self._sample_rate).item()
        )

        inference_time_taken = time.time() - now
        now = time.time()
        self._dispatch_event(speech_prob, resampled_frames)
        dispatch_time_taken = time.time() - now
        self._current_sample += merged_frame.samples_per_channel
        if self._current_sample % 100000 == 0:
            logging.debug(f"Processed {self._current_sample} samples, current merged frame size: {merged_frame.samples_per_channel}, time taken: {inference_time_taken:.3f}s, dispatch time taken: {dispatch_time_taken:.3f}s, speech prob: {speech_prob:.3f}")

    def _dispatch_event(self, speech_prob: int, original_frames: List[rtc.AudioFrame]):
        """
        Dispatches a VAD event based on the speech probability and the options
        Args:
            speech_prob: speech probability of the current frame
            original_frames: original frames of the current inference
        """

        samples_10ms = self._sample_rate / 100
        padding_count = int(
            self._padding_duration_samples // samples_10ms
        )  # number of frames to keep for the padding (one side)

        self._buffered_frames.extend(original_frames)
        if (
            not self._speaking
            and not self._waiting_start
            and len(self._buffered_frames) > padding_count
        ):
            self._buffered_frames = self._buffered_frames[
                len(self._buffered_frames) - padding_count :
            ]

        max_buffer_len = padding_count + max(
            self._max_buffered_samples // samples_10ms,
            self._min_speaking_samples // samples_10ms,
        )
        if len(self._buffered_frames) > max_buffer_len:
            # if unaware of this, may be hard to debug, so logging seems ok here
            logging.warning(
                f"VAD buffer overflow, dropping {len(self._buffered_frames) - max_buffer_len} frames"
            )
            self._buffered_frames = self._buffered_frames[
                len(self._buffered_frames) - max_buffer_len :
            ]

        if len(self._speech_probs) >0 and max(self._speech_probs) < self._threshold:
            self._speech_probs = []
        if speech_prob >= self._threshold:
            self._speech_probs.append(speech_prob)
        elif speech_prob < self._asume_speech_min_prob:
            self._speech_probs = []
        elif len(self._speech_probs) > 0:
            logging.debug(f"VAD: asuming speech, speech prob: {speech_prob:.3f}")
            self._speech_probs.append(speech_prob)
            speech_prob = self._threshold + 0.001
        if len(self._speech_probs) > self._asume_speech_max_count:
            self._speech_probs.pop(0)
        

        if speech_prob >= self._threshold:
            # speaking, wait for min_speaking_duration to trigger START_SPEAKING
            self._waiting_end = False
            if not self._waiting_start and not self._speaking:
                self._waiting_start = True
                self._start_speech = self._current_sample

            if self._waiting_start and (
                self._current_sample - self._start_speech >= self._min_speaking_samples
            ):
                logging.debug(
                    f"VAD: start speaking at {self._start_speech}, current sample: {self._current_sample - self._start_speech}"
                )
                self._waiting_start = False
                self._speaking = True
                event = agents.vad.VADEvent(
                    type=agents.vad.VADEventType.START_SPEAKING,
                    samples_index=self._start_speech,
                )
                self._event_queue.put_nowait(event)
                self.statusChanged(event.type)

                # since we're waiting for the min_spaking_duration to trigger START_SPEAKING,
                # the SPEAKING data is missing the first few frames, trigger it here
                # TODO(theomonnom): Maybe it is better to put the data inside the START_SPEAKING event?
                event = agents.vad.VADEvent(
                    type=agents.vad.VADEventType.SPEAKING,
                    samples_index=self._start_speech,
                    speech=self._buffered_frames[padding_count:],
                )

                return
            else:
                # still waiting for min_speaking_duration to trigger START_SPEAKING
                logging.debug(
                    f"VAD: still waiting for start speaking, current sample: {self._current_sample - self._start_speech}, current speech prob: {speech_prob:.3f}"
                )
                return

        if self._speaking:
            # we don't check the speech_prob here
            event = agents.vad.VADEvent(
                type=agents.vad.VADEventType.SPEAKING,
                samples_index=self._current_sample,
                speech=original_frames,
            )
            self._event_queue.put_nowait(event)
            self.statusChanged(event.type)

        if speech_prob < self._threshold:
            # stopped speaking, wait for min_silence_duration to trigger END_SPEAKING
            self._waiting_start = False
            if not self._waiting_end and self._speaking:
                self._waiting_end = True
                self._end_speech = self._current_sample

            if self._waiting_end and (
                self._current_sample - self._end_speech
                >= max(self._min_silence_samples, self._padding_duration_samples)
            ):
                self._waiting_end = False
                self._speaking = False
                event = agents.vad.VADEvent(
                    type=agents.vad.VADEventType.END_SPEAKING,
                    samples_index=self._end_speech,
                    duration=(self._current_sample - self._start_speech)
                    / self._sample_rate,
                    speech=self._buffered_frames,
                )
                self._event_queue.put_nowait(event)
                self.statusChanged(event.type)

    async def __anext__(self) -> agents.vad.VADEvent:
        if self._closed and self._event_queue.empty():
            raise StopAsyncIteration

        return await self._event_queue.get()
