# StyleGAN3 Remake of *Who The Fuck Is Conor?*

VIDEO LINK

The aim of this project was to experiment with ML video generation to create an interesting visual experiment, as well as to try my hand at algorithmic filmmaking for the first time. I just wanted to create a solidly watchable piece which is a bit more than a sketch, though not necesarilly polished.

## Workflow, other people's code and AI acknowledgements
I relied heavily on the Week 7 class Jupyter notebook called [](). I essentially wrapped the projection section from this file in a series of for loops. During the testing phase, where I was just working with 5-6 frames at a time, I was working directly in that notebook, so with the help of GitHub Copilot I put each sections from that section into its own loop. Then, since I wanted to create the whole film at a time, I manually wrapped those adapted sections in another greater for-loop, so that we could let it run overnight. I did this manually as I think the context window was a little too large for Copilot to get it right, and I really only had to copy/paste and change a couple of index variables etc.

I also used the [utils.py file]() from the same class, which I suppose was written by a lecturer. I added my own wrapper function for the , which was initially adapted from the Notebook by Copilot and which I then refactored into its own function `image_directory_to_tensors()`.
