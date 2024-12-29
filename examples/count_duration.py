import os
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt


def gtzan():

    root = "/datasets/gtzan"

    audios_dir = Path(root, "genres")

    audio_paths = sorted(list(Path(audios_dir).rglob("*.au")))

    durations = []

    for audio_path in audio_paths:
        duration = librosa.get_duration(path=audio_path)
        durations.append(duration)

    print("------ GTZAN ------")
    print("Files num: {}".format(len(durations)))
    print("Duration: {:.2f} s".format(np.sum(durations)))

    fig_path = "results/gtzan.pdf"
    plot_histogram(durations=durations, title="GTZAN", fig_path=fig_path)


def musdb18hq():

    root = "/datasets/musdb18hq"

    durations = []

    for split in ["train", "test"]:

        audio_names = sorted(os.listdir(Path(root, split)))

        for audio_name in audio_names:

            audio_path = Path(root, split, audio_name, "mixture.wav")
            duration = librosa.get_duration(path=audio_path)
            durations.append(duration)

    print("------ MUSDB18HQ ------")
    print("Files num: {}".format(len(durations)))
    print("Duration: {:.2f} s".format(np.sum(durations)))

    fig_path = "results/musdb18hq.pdf"
    plot_histogram(durations=durations, title="MUSDB18HQ", fig_path=fig_path)


def maestro():

    root = "/datasets/maestro-v3.0.0"

    audio_paths = sorted(list(Path(root).rglob("*.wav")))

    durations = []

    for audio_path in audio_paths:

        duration = librosa.get_duration(path=audio_path)
        durations.append(duration)

    print("------ MAESTRO V3.0.0 ------")
    print("Files num: {}".format(len(durations)))
    print("Duration: {:.2f} s".format(np.sum(durations)))

    fig_path = "results/maestro_v3.pdf"
    plot_histogram(durations=durations, title="MAESTRO V3.0.0", fig_path=fig_path)


def clotho():

    root = "/datasets/clotho"

    audio_paths = sorted(list(Path(root).rglob("*.wav")))

    durations = []

    for audio_path in audio_paths:

        duration = librosa.get_duration(path=audio_path)
        durations.append(duration)

    print("------ Clotho ------")
    print("Files num: {}".format(len(durations)))
    print("Duration: {:.2f} s".format(np.sum(durations)))

    fig_path = "results/clotho.pdf"
    plot_histogram(durations=durations, title="Clotho", fig_path=fig_path)


def audiocaps():

    root = "/datasets/audiocaps"

    audio_paths = sorted(list(Path(root).rglob("*.wav")))

    durations = []

    for audio_path in audio_paths:

        duration = librosa.get_duration(path=audio_path)
        durations.append(duration)

    print("------ AudioCaps ------")
    print("Files num: {}".format(len(durations)))
    print("Duration: {:.2f} s".format(np.sum(durations)))

    fig_path = "results/audiocaps.pdf"
    plot_histogram(durations=durations, title="AudioCaps", fig_path=fig_path)


def wavcaps():

    root = "/datasets/wavcaps"

    audio_paths = sorted(list(Path(root).rglob("*.flac")))

    durations = []
    sample_rates = []

    for audio_path in audio_paths:

        duration = librosa.get_duration(path=audio_path)

        if duration < 100000:
            durations.append(duration)

            sr = librosa.get_samplerate(path=audio_path)
            sample_rates.append(sr)

    print("------ WavCaps ------")
    print("Files num: {}".format(len(durations)))
    print("Duration: {:.2f} s".format(np.sum(durations)))

    fig_path = "results/wavcaps.pdf"
    plot_histogram(durations=durations, title="WavCaps", fig_path=fig_path)
    from IPython import embed; embed(using=False); os._exit(0)


def shutterstock():

    root = "/datasets/shutterstock/shutterstock_mp3"

    audio_paths = sorted(list(Path(root).rglob("*.flac")))

    durations = []
    sample_rates = []

    for audio_path in audio_paths:

        duration = librosa.get_duration(path=audio_path)

        if duration < 100000:
            durations.append(duration)

            sr = librosa.get_samplerate(path=audio_path)
            sample_rates.append(sr)

    print("------ Shutterstock ------")
    print("Files num: {}".format(len(durations)))
    print("Duration: {:.2f} s".format(np.sum(durations)))

    fig_path = "results/shutterstock.pdf"
    plot_histogram(durations=durations, title="Shutterstock", fig_path=fig_path)
    from IPython import embed; embed(using=False); os._exit(0)


def slakh2100():

    root = "/datasets/slakh2100_flac"

    audio_paths = sorted(list(Path(root).rglob("mix.flac")))

    durations = []
    sample_rates = []

    for audio_path in audio_paths:

        duration = librosa.get_duration(path=audio_path)
        durations.append(duration)

        sr = librosa.get_samplerate(path=audio_path)
        sample_rates.append(sr)

    print("------ Slakh2100 ------")
    print("Files num: {}".format(len(durations)))
    print("Duration: {:.2f} s".format(np.sum(durations)))

    fig_path = "results/slakh2100.pdf"
    plot_histogram(durations=durations, title="Slakh2100", fig_path=fig_path)
    from IPython import embed; embed(using=False); os._exit(0)


def isophonics():

    root = "/datasets/isophonics"

    audio_paths = sorted(list(Path(root).rglob("*.mp3")))

    durations = []
    sample_rates = []

    for audio_path in audio_paths:

        duration = librosa.get_duration(path=audio_path)
        durations.append(duration)

        sr = librosa.get_samplerate(path=audio_path)
        sample_rates.append(sr)

    print("------ Isophonics ------")
    print("Files num: {}".format(len(durations)))
    print("Duration: {:.2f} s".format(np.sum(durations)))

    fig_path = "results/isophonics.pdf"
    plot_histogram(durations=durations, title="Isophonics", fig_path=fig_path)
    

def voicebank_demand():

    root = "/datasets/voicebank-demand"

    audio_paths = sorted(list(Path(root).rglob("*.wav")))

    durations = []
    sample_rates = []

    for audio_path in audio_paths:

        duration = librosa.get_duration(path=audio_path)
        durations.append(duration)

        sr = librosa.get_samplerate(path=audio_path)
        sample_rates.append(sr)

    print("------ Isophonics ------")
    print("Files num: {}".format(len(durations)))
    print("Duration: {:.2f} s".format(np.sum(durations)))

    fig_path = "results/voicebank-demand.pdf"
    plot_histogram(durations=durations, title="Voicebank-Demand", fig_path=fig_path)
    from IPython import embed; embed(using=False); os._exit(0)


def plot_histogram(durations: list[float], title: str, fig_path: str):
    
    plt.figure()
    hist, bin_edges = np.histogram(durations, bins=20)
    width = 0.8 * (bin_edges[1] - bin_edges[0])
    plt.bar(x=bin_edges[:-1], height=hist, width=width)
    title += " Duration: {:.2f} h".format(np.sum(durations) / 3600)
    plt.title(title)
    plt.xlabel('Duration (s)')
    plt.ylabel('Audios Number')

    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path)
    print("Write out to {}".format(fig_path))


if __name__ == '__main__':

    # gtzan()
    # musdb18hq()
    # maestro()
    # clotho()
    # audiocaps()
    # wavcaps()
    # shutterstock()
    # slakh2100()
    # isophonics()
    voicebank_demand()