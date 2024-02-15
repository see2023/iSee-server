# 导入gradio和livekit的包
import gradio as gr
from livekit import rtc
from common.http_post import common_post
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

url_prefix = os.getenv('TEST_URL_PREFIX')
test_user = os.getenv('TEST_USER')
ws_url = os.getenv('LIVEKIT_URL')
print(url_prefix, test_user, ws_url)

# 定义一个函数，用于连接到LiveKit房间，并返回对方的音视频
async def join_room(ws_url, token):
    # 创建一个房间对象
    room = rtc.Room()
    # 连接到LiveKit服务器
    await room.connect(ws_url, token)
    # 定义一个全局变量，用于存储对方的音视频轨道
    global remote_audio, remote_video
    remote_audio = None
    remote_video = None
    # 定义一个事件处理器，用于订阅对方的音视频轨道
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        global remote_audio, remote_video
        print(f"track subscribed: {track}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            # 如果轨道是音频，就将它赋值给remote_audio
            remote_audio = track
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            # 如果轨道是视频，就将它赋值给remote_video
            remote_video = track
    # 等待房间连接成功
    # 返回对方的音视频轨道
    return remote_audio, remote_video

# 创建一个gradio接口，使用文本输入组件来输入ws_url和token，使用音频和图像输出组件来展示对方的音视频
async def main():
    data = common_post(url_prefix + '/api/v1/live/getToken', test_user, 'getToken')
    if data is not None:
        token = data['text']
    else:
        token = None
    demo = gr.Interface(
        join_room,
        [
            gr.inputs.Textbox(label="ws_url", default=ws_url),
            gr.inputs.Textbox(label="token", default=token)
        ],
        [
            gr.outputs.Audio(label="Remote Audio", type="numpy"),
            gr.outputs.Video(label="Remote Video", type="numpy")
        ]
    )
    demo.launch()



if __name__ == "__main__":
    asyncio.run(main())
