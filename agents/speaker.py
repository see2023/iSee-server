from dotenv import load_dotenv
load_dotenv()
import time
import torch
import numpy as np
from pyannote.audio import Model, Inference
from pyannote.core import Segment
from scipy.spatial.distance import cdist
import asyncio
import logging
import wave
import datetime
from dataclasses import dataclass
from typing import List, Dict
from common.config import config


@dataclass
class SpeakerEmbedding:
    duration:float
    embedding:np.ndarray


# 记录同一个人的多个音频embedding
class SpeakerEmbeddings:
    embeddings: List[SpeakerEmbedding]
    max_embeddings: int = 5
    average_embedding:np.ndarray = None
    average_distance:float = 0.0
    id:int = 0

    def __init__(self, id:int):
        self.id = id
        self.embeddings = []

    def add_embedding(self,  duration:float, embedding:np.ndarray):
        if len(self.embeddings) < self.max_embeddings:
            self.embeddings.append(SpeakerEmbedding(duration, embedding))
            logging.debug(f"Add new embedding with duration {duration} and shape {embedding.shape} for speaker {self.id}")
        else:
            # 计算新音频和现有音频的平均距离
            distances = []
            for e in self.embeddings:
                distances.append(cdist(embedding, e.embedding, metric="cosine")[0,0])
            average_distance = np.mean(distances)
            # 如果新音频距离平均距离小于当前平均距离，则替换掉最远的音频
            if average_distance < self.average_distance:
                max_index = np.argmax(distances)
                self.embeddings[max_index] = SpeakerEmbedding(duration, embedding)
                logging.debug(f"Replace embedding with duration {duration} and distance {average_distance} for speaker {self.id}, current average distance is {self.average_distance}")
            else:
                logging.debug(f"No replacement for speaker {self.id}, New embedding with duration {duration} and distance {average_distance} is far from average embedding with distance {self.average_distance}")
                return 
        self.update_average_embedding()       

    def update_average_embedding(self):
        if len(self.embeddings) == 0:
            return None
        # 取所有embedding的平均值作为最终的embedding
        embeddings = [e.embedding for e in self.embeddings]
        self.average_embedding = np.mean(embeddings, axis=0)
        # 计算所有embedding之间的平均距离
        if len(embeddings) == 1:
            self.average_distance = 0.0
            return
        distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                distances.append(cdist(embeddings[i], embeddings[j], metric="cosine")[0,0])
        self.average_distance = np.mean(distances)
        logging.debug(f"Update average distance to {self.average_distance} for speaker {self.id}")

    def get_embedding(self):
        return self.average_embedding
    
    # 判断是否是同一个人的声音，如果是，则判断距离来更新embeddings
    def is_same_speaker(self, embedding:np.ndarray, threshold:float = 0.25, duration:float = 0.0):
        if self.average_embedding is None or duration < 0.1:
            return False, None
        distance = cdist(self.average_embedding, embedding, metric="cosine")[0,0]
        if duration<5:
            # 对于短音频，将threshold按 时长与 5 的差距，线性差值到1.5倍
            threshold = threshold * ( 1 + 0.5 * (5 - duration)/(5 - config.agents.min_chunk_duration) )
        logging.debug(f"debug --- Distance between new embedding and average embedding for speaker {self.id}: {distance:.4f}, threshold: {threshold:.4f}, audio duration: {duration:.2f} seconds")
        if distance < threshold: 
            self.add_embedding(duration, embedding)
            return True, distance
        else:
            return False, distance
    
# https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
class Speaker:
    def __init__(self, device:str = "cpu", speaker_distance_threshold:float = 0.25, min_chunk_duration:float = 3.0, max_chunk_duration:float = 20.0):
        self.device = torch.device(device)
        try:
            self.model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        except Exception as e:
            logging.error(f"Error loading pyannote/wespeaker-voxceleb-resnet34-LM model: {e}")
            exit()
        self.inference = Inference(self.model, window="whole")
        self.inference.to(self.device)
        self.speaker_distance_threshold = speaker_distance_threshold
        self.min_chunk_duration = min_chunk_duration
        self.max_chunk_duration = max_chunk_duration
        self.speakers: Dict[int, SpeakerEmbeddings] = {}
        self.last_speaker_id = 0

    def get_embedding_by_file(self, file_path:str) -> np.ndarray:
        embedding = self.inference(file_path)
        # numpy.ndarray (float32,256)
        embedding = embedding.reshape(1, -1)
        return embedding
    
    # {"waveform": array or tensor, "sample_rate": int}
    def get_embedding_from_buffer(self, buf, sample_rate:int):
        total_duration = len(buf) / sample_rate
        if total_duration < self.min_chunk_duration:
            logging.debug(f"Audio buffer is too short, duration: {total_duration:.2f} seconds, min_chunk_duration: {self.min_chunk_duration:.2f} seconds")
            return None
        if config.agents.speaker_write_wav:
            wave_file = f'records/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.wav'
            with wave.open(wave_file, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(sample_rate)
                wav.writeframes(buf)
            logging.debug(f"Saved buffer to {wave_file}")
        audio_data = np.frombuffer(buf, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        audio_data = audio_data.reshape(1, -1)
        audio_tensor = torch.from_numpy(audio_data)
        seg = Segment(0, min(total_duration, self.max_chunk_duration))
        embedding = self.inference.crop({"waveform": audio_tensor, "sample_rate": sample_rate}, chunk=seg)
        embedding = embedding.reshape(1, -1)
        return embedding
    
    def get_speakerid_from_buffer(self, buf, sample_rate:int):
        try:
            start_time = time.time()
            new_embedding = self.get_embedding_from_buffer(buf, sample_rate)
            total_duration = len(buf) / sample_rate
            if new_embedding is None:
                return 0
            # check if self.speakers is empty
            if not self.speakers or len(self.speakers) == 0:
                self.last_speaker_id = 1
                self.speakers[self.last_speaker_id] = SpeakerEmbeddings(self.last_speaker_id)
                self.speakers[self.last_speaker_id].add_embedding(total_duration, new_embedding)
                logging.info(f"First speaker added with id: {self.last_speaker_id}, audio duration: {total_duration:.3f} seconds")
                return self.last_speaker_id
            for speaker_id, speaker_embedding in self.speakers.items():
                rt, distance = speaker_embedding.is_same_speaker(new_embedding, self.speaker_distance_threshold, total_duration)
                if rt:
                    logging.info(f"Speaker found with id: {speaker_id}, time taken: {time.time() - start_time:.4f} seconds, min_distance: {distance:.4f}, audio duration: {total_duration:.3f} seconds")
                    return speaker_id
            self.last_speaker_id += 1
            self.speakers[self.last_speaker_id] = SpeakerEmbeddings(self.last_speaker_id)
            self.speakers[self.last_speaker_id].add_embedding(total_duration, new_embedding)
            logging.debug(f"New speaker added with id: {self.last_speaker_id}, time taken: {time.time() - start_time:.4f} seconds, audio duration: {total_duration:.3f} seconds")
            return self.last_speaker_id
        except Exception as e:
            logging.error(f"Error getting speaker id from buffer: {e}")
            return 0

    async def get_speakerid_from_buffer_async(self, buf, sample_rate:int):
        return await asyncio.get_event_loop().run_in_executor(None, self.get_speakerid_from_buffer, buf, sample_rate)

    def get_distance_by_file(self, file_path_1:str, file_path_2:str):
        embedding_1 = self.get_embedding_by_file(file_path_1)
        embedding_2 = self.get_embedding_by_file(file_path_2)
        distance = cdist(embedding_1, embedding_2, metric="cosine")[0,0]
        return distance

def test_get_speaker_embedding():
    file_path_1 = 'output.wav'
    file_path_2 = 'output_local.wav'
    file_path_3 = '01.wav'
    file_path_4 = '02.wav'

    file_path_3 = 'records/2024-04-01-11-57-59.wav'
    file_path_4 = 'records/2024-04-01-11-58-10.wav'

    speaker = Speaker()
    start_time = time.time()
    distance = speaker.get_distance_by_file(file_path_1, file_path_2)
    logging.info(f'Distance between two embeddings: {distance:.4f}')
    end_time = time.time()
    logging.info(f"Time taken for inference: {end_time - start_time:.4f} seconds")
    start_time = time.time()
    distance = speaker.get_distance_by_file(file_path_3, file_path_4)
    logging.info(f'Distance between two embeddings: {distance:.4f}')
    end_time = time.time()
    logging.info(f"Time taken for inference: {end_time - start_time:.4f} seconds")





if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
    )
    test_get_speaker_embedding()
