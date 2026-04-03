import yt_dlp

import os
import yt_dlp

def download_from_url(path, url):
    proxy = os.environ.get("HTTP_PROXY")  # or "HTTPS_PROXY"

    ydl_opts = {
        'format': 'bestaudio/best',
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': f'{path}.%(ext)s',
        'proxy': proxy  # <-- use env variable here
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def get_video_id(url):
    ydl_opts = {
        'quiet': True,  # Suppress console output
        'no_warnings': True,
        'force_generic_extractor': True, # Ensures extraction works for various URLs
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract info without downloading
        info_dict = ydl.extract_info(url, download=False)

        # The 'id' field contains the video ID
        video_id = info_dict.get("id", None)
        return video_id