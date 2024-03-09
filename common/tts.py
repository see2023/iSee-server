import aiohttp
import os,sys
import logging
import asyncio
import json
import msgpack
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from common.config import config

import xml.etree.cElementTree as ET
import azure.cognitiveservices.speech as speechsdk


'''
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice#use-speaking-styles-and-roles

<speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xmlns:emo="http://www.w3.org/2009/10/emotionml" version="1.0" xml:lang="zh-CN">
  <voice name="en-US-AriaNeural">
    <mstts:viseme type="FacialExpression"/>
    <mstts:express-as style="cheerful">
      <prosody rate="0%" pitch="0%" volume="0%">Tom usually ride a bike in Saturday afternoon.</prosody>
    </mstts:express-as>
  </voice>
</speak>


'''

class AzureTTS:
    def __init__(self) -> None:
        # read from env
        self.subscription_key = os.environ.get('AZURE_TTS_KEY', None)
        self.region = os.environ.get('AZURE_TTS_REGION', 'eastasia')
        self.BlendShapes = []
    
    def viseme_cb(self, evt):
        # logging.debug("Viseme event received: audio offset: {}ms, viseme id: {}.".format(
        #     evt.audio_offset / 10000, evt.viseme_id))
        try:
            animation = json.loads(evt.animation)
            self.BlendShapes = self.BlendShapes + animation['BlendShapes']
        except:
            # print('json parse error')
            pass


    def text_to_speech_and_visemes(self, text, voice_name='zh-CN-YunxiNeural', output_format='riff-24khz-16bit-mono-pcm', role=' ', style='cheerful', rate='0%', pitch='0%', volume='0%'):
        # create xml
        self.BlendShapes = []
        root = ET.Element('speak')
        root.set('xmlns', 'http://www.w3.org/2001/10/synthesis')
        root.set('xmlns:mstts', 'http://www.w3.org/2001/mstts')
        root.set('xmlns:emo', 'http://www.w3.org/2009/10/emotionml')
        root.set('version', '1.0')
        root.set('xml:lang', 'zh-CN')
        voice = ET.SubElement(root, 'voice')
        voice.set('name', voice_name)
        viseme = ET.SubElement(voice, 'mstts:viseme')
        viseme.set('type', 'FacialExpression')
        express_as = ET.SubElement(voice, 'mstts:express-as')
        express_as.set('style', style)
        if role and len(role) > 1:
            express_as.set('role', role)
        prosody = ET.SubElement(express_as, 'prosody')
        prosody.set('rate', rate)
        prosody.set('pitch', pitch)
        prosody.set('volume', volume)
        prosody.text = text
        tree = ET.ElementTree(root)
        xml = ET.tostring(root, encoding='utf-8', method='xml').decode('utf-8')
        logging.debug(xml)
        speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        speech_synthesizer.viseme_received.connect(self.viseme_cb)
        result = speech_synthesizer.speak_ssml_async(xml).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logging.debug("Speech synthesized to speaker for text [ " + text + " ], audio length: "
                + str(len(result.audio_data)) + " bytes, visemes: " + str(len(self.BlendShapes)) + " frames")
            self.BlendShapes = np.array(self.BlendShapes)
            self.BlendShapes = (self.BlendShapes * 1000).astype(np.int16)
            return {'audio': result.audio_data, 'visemes': self.BlendShapes.tolist()}
        else:
            logging.error("Speech synthesis canceled, result.reason: " + result.reason + ', detail: ' + result.error_details)
            return None
    
    async def text_to_speech_and_visemes_async(self, text, voice_name='zh-CN-YunxiNeural', output_format='riff-24khz-16bit-mono-pcm', role=' ', style='cheerful', rate='0%', pitch='0%', volume='0%'):
        return await asyncio.get_event_loop().run_in_executor(None, self.text_to_speech_and_visemes, text, voice_name, output_format, role, style, rate, pitch, volume)

async def text_to_speech_and_visemes_local(text, text_language='zh', top_k=10, top_p=1, temperature=1, url='http://192.168.17.141:9880/'):
    try:
        async with aiohttp.ClientSession() as session:
            logging.debug(f'Sending text to {url}: {text}')
            async with session.post(url, json={'text': text, 'text_language': text_language, 'top_k': top_k, 'top_p': top_p, 'temperature': temperature}) as response:
                if response.status == 200:
                    # read binary raw response
                    raw_response = await response.read()
                    logging.debug(f'Got response from {url}, length: {len(raw_response)}')
                    msgpack_response = msgpack.unpackb(raw_response)
                    return msgpack_response
                else:
                    logging.error(f'text_to_speech_and_visemes_local status Error: {response.status} {response.reason}')
                    return None
    except Exception as e:
        logging.error(f'text_to_speech_and_visemes_local Error: {e}')
        return None

async def text_to_speech_and_visemes(text):
    tts_type = config.llm.tts
    if tts_type == 'azure':
        tts = AzureTTS()
        res =  await tts.text_to_speech_and_visemes_async(text, voice_name=config.llm.voice_name)
        return res, 60.0
    elif tts_type == 'vits':
        res = await text_to_speech_and_visemes_local(text, url=config.llm.vits_url)
        return res, 86.1328125
    else:
        # logging.error(f'Unsupported TTS type: {tts_type}')
        return None, 0.0

def get_visemes_fps():
    return 86.1328125 if config.llm.tts == 'vits' else 60

async def test_async():
    tts = AzureTTS()
    text = '你好，欢迎使用语音合成服务。'
    result = await tts.text_to_speech_and_visemes_async(text)
    if result:
        with open('output.wav', 'wb') as f:
            f.write(result['audio'])
        visemes = np.array(result['visemes'])
        visemes = visemes / 1000
        logging.info(f'visemes shape: {visemes.shape}')
        np.save('visemes.npy', visemes)
        logging.info(f'got audio and visemes, saved to output.wav and visemes.npy')
    
    result_local = await text_to_speech_and_visemes_local(text)
    if result_local:
        with open('output_local.wav', 'wb') as f:
            f.write(result_local['audio'])
        visemes_local = np.array(result_local['visemes'])
        # values / 1000
        visemes_local = visemes_local / 1000
        logging.info(f'visemes_local shape: {visemes_local.shape}')
        np.save('visemes_local.npy', visemes_local)
        logging.info(f'got audio and visemes_local, saved to output_local.wav and visemes_local.npy')



if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s [in %(pathname)s:%(lineno)d] - %(message)s',
    )
    asyncio.run(test_async())
