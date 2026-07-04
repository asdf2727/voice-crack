from audio.device_stream import *
from datasets.vctk import VCTKDataset

from torch.utils.data import DataLoader

def main():
    ds = VCTKDataset("../datasets/VCTK-Corpus-0.92")
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    #source = FileSource("../glad.flac", 0.02)
    source = DeviceSource(0.02)
    #FileSink.dump_source("../output.wav", source, max_time=10)
    DeviceSink.dump_source(source)

if __name__ == "__main__":
    main()