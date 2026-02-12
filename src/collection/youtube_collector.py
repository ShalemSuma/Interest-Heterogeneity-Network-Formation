"""
YouTube Data Collector

Collects channel metadata and video data from the YouTube Data API v3.
Credentials are loaded from environment variables (see .env.example).
"""

import os
import csv
import json
import time
import pickle
from datetime import timedelta, datetime

import pandas as pd
import matplotlib.pyplot as plt

try:
    import googleapiclient.discovery
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
except ImportError:
    pass  # Allow import even without google libraries installed

from ..utils.config import get_data_path, get_youtube_credentials

SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]


class YouTubeDataCollector:
    """Collect YouTube channel and video data via the Data API v3."""

    def __init__(
        self,
        input_file=None,
        output_file="youtube_channel_data.csv",
        video_output_dir="video_data",
        max_videos=200,
        quota_limit=9000,
        checkpoint_file="youtube_checkpoint.json",
    ):
        self.input_file = input_file or str(get_data_path("raw") / "YoutubeNodes.csv")
        self.output_file = output_file
        self.video_output_dir = video_output_dir
        self.max_videos = max_videos
        self.quota_limit = quota_limit
        self.checkpoint_file = checkpoint_file

        self.quota_used = 0
        self.start_time = None
        self.all_channel_data = []
        self.processed_indices = []

        os.makedirs(self.video_output_dir, exist_ok=True)
        self.load_checkpoint()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def get_authenticated_service(self):
        """Authenticate via OAuth2 (uses env vars, not hardcoded secrets)."""
        creds = None
        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as f:
                creds = pickle.load(f)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                yt_creds = get_youtube_credentials()
                client_config = {
                    "installed": {
                        "client_id": yt_creds["client_id"],
                        "client_secret": yt_creds["client_secret"],
                        "project_id": yt_creds["project_id"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": ["http://localhost"],
                    }
                }
                flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
                creds = flow.run_local_server(port=0)

            with open("token.pickle", "wb") as f:
                pickle.dump(creds, f)

        return googleapiclient.discovery.build("youtube", "v3", credentials=creds)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def fix_channel_id(channel_id):
        if channel_id.startswith("--"):
            return "UC" + channel_id
        if not channel_id.startswith("UC"):
            return "UC" + channel_id
        return channel_id

    def get_channel_videos(self, youtube, channel_id, max_results=None):
        quota_used = 0
        try:
            fixed = self.fix_channel_id(channel_id)
            resp = youtube.channels().list(part="contentDetails", id=fixed).execute()
            quota_used += 1
            if not resp.get("items"):
                return [], quota_used

            playlist_id = resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            video_ids, token = [], None

            while True:
                resp = youtube.playlistItems().list(
                    part="snippet", playlistId=playlist_id,
                    maxResults=50, pageToken=token,
                ).execute()
                quota_used += 1

                for item in resp.get("items", []):
                    video_ids.append(item["snippet"]["resourceId"]["videoId"])
                    if max_results and len(video_ids) >= max_results:
                        return video_ids, quota_used

                token = resp.get("nextPageToken")
                if not token:
                    break
                time.sleep(0.1)

            return video_ids, quota_used
        except Exception as e:
            print(f"Error getting videos for {channel_id}: {e}")
            return [], quota_used

    def get_video_details(self, youtube, video_ids, batch_size=50):
        details, quota_used = [], 0
        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i: i + batch_size]
            try:
                resp = youtube.videos().list(
                    part="snippet,contentDetails", id=",".join(batch)
                ).execute()
                quota_used += 1

                for item in resp.get("items", []):
                    dur = item["contentDetails"]["duration"][2:]
                    h = m = s = 0
                    if "H" in dur:
                        h, dur = int(dur.split("H")[0]), dur.split("H")[1]
                    if "M" in dur:
                        m, dur = int(dur.split("M")[0]), dur.split("M")[1]
                    if "S" in dur:
                        s = int(dur.split("S")[0])
                    total = h * 3600 + m * 60 + s

                    details.append({
                        "video_id": item["id"],
                        "title": item["snippet"]["title"],
                        "duration_seconds": total,
                        "duration_formatted": str(timedelta(seconds=total)),
                        "published_at": item["snippet"]["publishedAt"],
                    })
                time.sleep(0.1)
            except Exception as e:
                print(f"Error getting video details: {e}")

        return details, quota_used

    def process_channel(self, youtube, channel_name, channel_id, max_videos=None):
        fixed = self.fix_channel_id(channel_id)
        quota_used = 0

        try:
            resp = youtube.channels().list(part="statistics", id=fixed).execute()
            quota_used += 1
            total_avail = int(resp["items"][0]["statistics"].get("videoCount", 0)) if resp.get("items") else 0
        except Exception:
            total_avail = 0

        vids, q = self.get_channel_videos(youtube, fixed, max_videos)
        quota_used += q

        if not vids:
            return {"channel_name": channel_name, "channel_id": fixed,
                    "status": "no_videos", "video_count": 0, "videos": []}, quota_used

        details, q = self.get_video_details(youtube, vids)
        quota_used += q

        return {
            "channel_name": channel_name,
            "channel_id": fixed,
            "status": "success",
            "video_count": len(details),
            "total_available_videos": total_avail,
            "has_more_videos": max_videos and len(vids) >= max_videos,
            "total_duration_seconds": sum(v["duration_seconds"] for v in details),
            "videos": details,
        }, quota_used

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self):
        with open(self.checkpoint_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "processed_indices": self.processed_indices,
                "channel_data": self.all_channel_data,
                "quota_used": self.quota_used,
            }, f, indent=2, default=str)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            return
        try:
            with open(self.checkpoint_file) as f:
                cp = json.load(f)
            self.all_channel_data = cp.get("channel_data", [])
            self.processed_indices = cp.get("processed_indices", [])
            self.quota_used = cp.get("quota_used", 0)
            print(f"Checkpoint: {len(self.processed_indices)} channels, quota {self.quota_used}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main collection loop
    # ------------------------------------------------------------------
    def collect_data(self):
        df = pd.read_csv(self.input_file)
        id_col = "Id" if "Id" in df.columns else "id"
        label_col = "Label" if "Label" in df.columns else "label"

        youtube = self.get_authenticated_service()
        self.start_time = time.time()
        remaining = [i for i in range(len(df)) if i not in self.processed_indices]

        try:
            for idx in remaining:
                if self.quota_used >= self.quota_limit:
                    print(f"Quota limit reached ({self.quota_used})")
                    break
                row = df.iloc[idx]
                data, q = self.process_channel(
                    youtube, str(row[label_col]), str(row[id_col]), self.max_videos
                )
                self.all_channel_data.append(data)
                self.processed_indices.append(idx)
                self.quota_used += q
                self.save_checkpoint()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nInterrupted — saving checkpoint...")
            self.save_checkpoint()

        return self.all_channel_data
