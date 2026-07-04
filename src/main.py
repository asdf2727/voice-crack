from audio.device_stream import *

def main():
    #source = FileSource("../glad.flac", 0.02)
    source = DeviceSource(0.02)
    #FileSink.dump_source("../output.wav", source, max_time=10)
    DeviceSink.dump_source(source)

if __name__ == "__main__":
    main()