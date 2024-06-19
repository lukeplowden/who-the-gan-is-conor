25fps
15 seconds intro, then at 33:05 there is another section of titles until 35:12

01:02:10 until 01:08:08

# Steps
## Video editing:
1. facetracking (approximate)
2. cut detection.
   - firstly, just cut the title sequences and the rest of the film into two big chunks to use for training 2 models. then with the main section of the film...
   - convert these chunks for training into images
3. export each cut into a separate mp4 file
4. convert each file into a series of images
5. structure the files so that each sequence has its own folder.

## Training
Train two models of StyleGAN3-T:
- first one from the frames in the title and second from the frames in the main parts of the film

## Projection
- Taking 1 frame every second, project this into the model.
- Interpolate 25 frames between each new image, and stitch the whole thing back together.
- We maintain the original cuts because we do not interpolate between cuts. This does add a problem however that the final frame of a sequence does not last a full second, which I will have to find a way around to keep the same runtime.