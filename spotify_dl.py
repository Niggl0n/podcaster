import logging

from savify import Savify
from savify.types import Type, Format, Quality
from savify.utils import PathHolder
# Quality Options: WORST, Q32K, Q96K, Q128K, Q192K, Q256K, Q320K, BEST
# Format Options: MP3, AAC, FLAC, M4A, OPUS, VORBIS, WAV
s = Savify(api_credentials=("6159d2c3b1ca455898d94b26ceebd659", "3f741a82ca324119b7d08ef14c20c6d5"),
           quality=Quality.BEST,
           download_format=Format.MP3,
           path_holder=PathHolder(downloads_path='~/Music'),
           # group='%artist%/%album%',
           skip_cover_art=False,
           logger=logging
            )

# s = Savify(api_credentials=("6159d2c3b1ca455898d94b26ceebd659","3f741a82ca324119b7d08ef14c20c6d5"))
s.download("https://open.spotify.com/playlist/4p0VhbO477w5jqd32fMLNV?si=f377fd8794814f27")

print("Done!")

