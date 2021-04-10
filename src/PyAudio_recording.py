"""
Preparation - download pyaudio
    In Windows : https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
        Teaching : 
            https://www.twblogs.net/a/5c4b20bcbd9eee6e7d81e2bf
            https://medium.com/@jscinin/python-3-7-install-pyaudio-e7c75edbbef4
    
    In Linux : Using terminal to do the following commands.
        sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
        pip3 install pyaudio

My environment
    OS : Ubuntu 20.04.1 LTS  (64bit)
    Editor : Visual Studio Code
    Python version : Python 3.8.2

----------------------------------------------------------------------------

modules
    pyaudio - can play and record audio.
    wave - read and write .wav files.

References
    pyaudio documentation : 
        https://people.csail.mit.edu/hubert/pyaudio/docs/

    pyaudio (Chinese teaching) : 
        https://zwindr.blogspot.com/2017/03/python-pyaudio.html

    StackOverflow (Explanation about this example) : 
        https://stackoverflow.com/questions/35970282/what-are-chunks-samples-and-frames-when-using-pyaudio
"""
import pyaudio
import wave

def recording():
    pass

def main():
    """ 
    @chunk : The number of frames in the buffer.
    
    @FORMAT : pyaudio.paInt16 : A special parameter in pyaudio. (?

            By calling the function pyaydio.get_sample_size(),
            size of each sample will be 2 bytes.

            So, size of each frame is 4 bytes.
    
    @CHANNELS : a parameter, if it be assigned to...
            1 : Mono (單聲道)
            2 : Stereo (雙聲道) -> Each frame will have 2 samples.
    
    @RATE : The number of samples collected per second.

    @RECORD_SECONDS : The length of time of the output file. (unit : second)
    """
    chunk = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 3

    while True:
        recording = input("(1)錄音並存檔\n(2)離開\nInput : ")

        if recording == '2':
            break

        # Declaring an instance of PyAudio. (PyAudio is a class.)
        p = pyaudio.PyAudio()

        # Using open method of p to create an instance of stream.
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=chunk)

        print("---recording---")

        # Creating an object of list.
        frames = []

        # ? ? ?
        print((RATE/chunk) * RECORD_SECONDS)

        for i in range(0, (RATE//chunk * RECORD_SECONDS)):
            # By read method, it gets audio data from microphone.
            data = stream.read(chunk)   # data will be a segment of audio.
            frames.append(data)         # Let the segment be put into a list.

        print("---done recording---")

        stream.close()

        # Closing PyAudio.
        p.terminate()

        # Let the data of audio be written into a .wav file.

        # Output file name.
        WAVE_OUTPUT_FILENAME = input('Input the file name, it will append \'.wav\' dynamically : ')
        WAVE_OUTPUT_FILENAME += ".wav"

        # Here opens the file and records the all audio data.
        # Using binary mode to record the audio data.
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')

        """
        Following methods let ...
            1.  channel be stereo,
            2.  width of a sample be 2 bytes, and
            3.  frame rate be 2.
        Finally, wf writes the audio data into the file by binary then closing the file.
        """
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("---------------")

if __name__ == '__main__':
    main()