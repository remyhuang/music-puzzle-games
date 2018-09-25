# Music Puzzle Games
TensorFlow implementation of [Generating Music Medleys via Playing Music Puzzle Games](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16174)
* Similarity Embedding Network (SEN) trained on downbeat-informed data

**Please cite this paper if this code/work is helpful:**

    @inproceedings{huang2018generating,
      title={Generating music medleys via playing music puzzle games},
      author={Huang, Yu-Siang and Chou, Szu-Yu and Yang, Yi-Hsuan},
      booktitle={AAAI},
      pages={2281--2288},
      year={2018}
    }
    
## Environment
* Python 3.6
* TensorFlow 1.2.0
* NumPy 1.14.3
* LibROSA 0.6.2
* Pydub 0.22.1

There are already three sample audio clips in the `data` folder. You can directly run the code.

    $ git clone https://github.com/remyhuang/music-puzzle-games.git
    $ cd music-puzzle-games
    $ python main.py
   
## Use
Replace the sample audio files (`mp3 format`) in the `data` for your own purpose.
(Noted: Due to using the brute-force method to find the best permutation, the total number of audio files __should not exceed 11__.)

## Output
Three default output files:
* __best_permutation.txt__: the best permutation calculated by pairwise similarities
* __output.csv__: pairwise similarities 
* __output.mp3__: concatenated audio file of the best permutation

## Contact
Please feel free to contact [Yu-Siang Huang](https://remyhuang.github.io/) if you have any questions.