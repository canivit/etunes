# etunes

etunes is an AI application that detects user's emotion from their face and plays a song in
Spotify that matches that emotion.

The application uses two different neural networks trained with Pytorch to classify faces and
songs.

[Watch demo video](https://youtu.be/PXW7wj1GIMA)

![eTunes Demo](https://etunes-public.s3.us-east-2.amazonaws.com/etunes.png)

## Setting Up The Environment

eTunes is a Python project and the provided [`requirements.txt`](requirements.txt) file can be
used to install dependencies.

Alternatively, if you use [nix](https://nixos.org/), [flake.nix](flake.nix) can be used to
create a reproducible dev environment.

## Running The Application

Download trained face model [here](https://etunes-public.s3.us-east-2.amazonaws.com/face_model.pt).

Download trained song model [here](https://etunes-public.s3.us-east-2.amazonaws.com/song_model.pt)

After model files are downloaded, the application script can be run as:

```bash
python etunes.py --face-model face_model.pt --song-model song_model.pt
```

## Face Classification

We trained a CNN with 3 convolutional layers on the 
[FER2013](https://paperswithcode.com/dataset/fer2013) dataset.
We removed **disgust** and **surprise** emotions from the original dataset since it would be
difficult to find songs that would match to those emotions. As a result, the model can classify a
face as one of these emotions:

- Anger
- Fear
- Happy
- Sad
- Neutral

Scripts for training and evaluating the face classification model can be found in [face](./face)
folder. We trained the face model in AWS Sagemaker. [run_sagemaker.py](./face/run_sagemaker.py)
script can be used to launch the training script in Sagemaker and monitor the training process.

With learning rate of **0.001** and **500** epochs, the model achieved the following performance
on the test set:

| Precision | Recall | F1     | Accuracy |
| --------- | ------ | ------ | -------- |
| 60.63%    | 60.78% | 60.70% | 63.18%   |

In practice, we found this performance to be high enough for the application to be
robust in good lighting.

## Song Classification

### Dataset

We used [Spotify Web API](https://developer.spotify.com/documentation/web-api) to create our
own dataset. Spotify API provides the following numeric features for each song:

- Acousticness
- Danceability
- Energy
- Instrumentalness
- Liveness
- Speechiness
- Tempo
- Valence

We chose 4 different playlists on Spotify for each of the emotions: anger, fear, happy, sad.
Then, we used [prep_data.py](music/prep_data.py) script to fetch the numeric features of each song,
assign an emotion label, and save it to a csv file.

## Training

Scripts for training and evaluating the song classification model can be found in [music](./music)
folder.

With learning rate of **0.001** and **100** epochs, the model achieved the following performance
on the test set:

| Precision | Recall | F1     | Accuracy |
| --------- | ------ | ------ | -------- |
| 86.85%    | 84.58% | 85.42% | 85.06%   |

## Collaborators

- [Can Ivit](https://www.linkedin.com/in/canivit/)
- [Nileena John](https://www.linkedin.com/in/nileena-john/)
- [Teera Tesharo](https://www.linkedin.com/in/teera74/)
