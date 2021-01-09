import sys
sys.path.append('/data7/MEmoBert')
from preprocess.tasks.common import VideoCutter
from preprocess.tasks.text import TranscriptPackager


start = 1000
test_num = 10
package_transcript = TranscriptPackager('test_transcript_save')
ass_path = '/data7/emobert/data_nomask/transcripts/raw/No0001.The.Shawshank.Redemption.ass'
transcripts = package_transcript(ass_path)
for i, content in enumerate(transcripts[start:start+test_num]):
    print(i, content)

cut_video = VideoCutter('test_video_cut')
cut_video("/data7/emobert/resources/raw_movies/No0001.The.Shawshank.Redemption.mp4", transcripts[start:start+test_num])
