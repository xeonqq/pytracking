## Example

```bash
conda activate pytracking
python3 run_video.py dimp dimp50 ~/Videos/2025-02-02-ehrwld/funpark_cut.MP4 --save_results
python3 center_stage.py --input_video ~/Videos/2025-02-02-ehrwld/funpark_cut.MP4  --bbox_file /home/qq/repo/pytracking/pytracking/tracking_results/dimp/dimp50/video_funpark_cut_1.txt --output_video out.mp4 --smooth_alpha 0.1
```

## Troubleshooting
In case of hanging in program, delete $HOME/.cache/torch_extensions/py312_cu121/_prroi_pooling/lock

## Video
[![video original](https://img.youtube.com/vi/nAIGv-2Wb0U/0.jpg)](https://youtu.be/nAIGv-2Wb0U)
[![video after center stage](https://img.youtube.com/vi/PjbBjZj1J0c/0.jpg)](https://youtu.be/PjbBjZj1J0c)
