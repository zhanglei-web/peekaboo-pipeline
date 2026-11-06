"""Read-only triplet content matching against each exact capture; preserve labels."""
from pathlib import Path
import sys,json,fcntl,traceback
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/baai/zl/Qwen3.8-27B/scripts')
from run_scene_event_batch import sha,atomic,now
cv2.setNumThreads(1)

def check(row):
 assert sha(Path(row['target_markdown']))==row['target_markdown_sha256']
 assert sha(Path(row['source'])/'meta.json')==row['metadata_sha256']
 video=cv2.VideoCapture(str(Path(row['source'])/'videos/left.mkv'));checks=[]
 try:
  wanted={s['original_video_frame_index']:s for e in row['event_index']['events'] for s in e['states']}
  thumbnails=[]
  for f in range(min(row['source_frame_count'],max(wanted)+91)):
   ok,image=video.read()
   if not ok:raise ValueError('Raw video decode failed at '+str(f))
   thumbnails.append(cv2.resize(image,(160,100),interpolation=cv2.INTER_AREA))
  thumbnails=np.asarray(thumbnails)
  for frame,state in sorted(wanted.items()):
   assert sha(Path(state['image_path']))==state['image_sha256']
   image=cv2.imread(state['image_path']);assert image is not None
   image=cv2.resize(image,(160,100),interpolation=cv2.INTER_AREA)
   lo,hi=max(0,frame-90),min(len(thumbnails),frame+91)
   errors=np.abs(thumbnails[lo:hi].astype(float)-image).mean(axis=(1,2,3));nearest=lo+int(np.argmin(errors));mae=float(errors[nearest-lo]);corr=float(np.corrcoef(thumbnails[nearest].reshape(-1),image.reshape(-1))[0,1])
   checks.append({'filename_frame_index':frame,'best_matching_decoded_frame_index':nearest,'decoded_minus_filename':nearest-frame,'thumbnail_mean_absolute_pixel_error':mae,'thumbnail_correlation':corr})
   if not np.isfinite(corr) or corr<.99 or mae>4:raise ValueError('Event image does not match this raw video: f'+str(frame))
  for e in row['right_event_index']['events']:
   for s in e['states']:
    assert sha(Path(s['image_path']))==s['image_sha256']
    assert cv2.imread(s['image_path']) is not None
 finally:video.release()
 return checks

def main():
 with (ROOT/'input_audit.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  rows=json.loads((ROOT/'annotation_inventory.json').read_text())['episodes']
  report={'status':'running','started_at':now(),'episodes':{},'policy':'Same raw-video content matching thresholds as 11.3; preserve filename frame labels even if best decoded index differs.'}
  for row in rows:
   try:
    checks=check(row);report['episodes'][row['episode_id']]={'status':'passed','checks':checks,'frame_index_drift_count':sum(x['decoded_minus_filename']!=0 for x in checks)}
   except Exception as exc:report['episodes'][row['episode_id']]={'status':'needs_repair','error':repr(exc)}
   atomic(ROOT/'input_audit.json',report)
  report.update(status='passed' if all(x['status']=='passed' for x in report['episodes'].values()) else 'needs_repair',finished_at=now())
  atomic(ROOT/'input_audit.json',report)
if __name__=='__main__':main()
