## Example

```bash
conda activate pytracking
python3 run_video.py dimp dimp50 ~/Videos/2025-02-02-ehrwld/funpark_cut.MP4 --save_results
python3 center_stage.py --input_video ~/Videos/2025-02-02-ehrwld/funpark_cut.MP4  --bbox_file /home/qq/repo/pytracking/pytracking/tracking_results/dimp/dimp50/video_funpark_cut_1.txt --output_video out.mp4 --smooth_alpha 0.1
```
