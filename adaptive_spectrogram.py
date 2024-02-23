from scipy import signal
from scipy.fft import fft, fftshift, ifft
import numpy as np
from scipy.stats import pearsonr

WS_SMALL = 2048
WS_LARGE = 4096
HS_SMALL = int(WS_SMALL / 2)
HS_LARGE = int(WS_LARGE / 2)
TOL = 1e-3


class AdaptiveSpectrogram:

    def __init__(self, x, corr_threshold, logger):
        """Constructor of the class AdaptiveSpectrogram

        Parameters:
        x : signal for which the adaptive spectrogram needs to be computed
        corr_threshold : threshold for correlation coefficient between adjacent frames
        logger : for logging debug messages

        Returns:
        None
        """
        self.x = x
        self.x_padded = x
        self.corr_threshold = corr_threshold
        self.logger = logger
        self.smallWindow = np.sqrt(signal.windows.hann(WS_SMALL))
        self.largeWindow = np.sqrt(signal.windows.hann(WS_LARGE))

    def pad_signal(self):
        """Function to pad the signal with zeros

        Parameters:
        None

        Returns:
        None
        """
        pad_left = WS_SMALL - int(HS_SMALL)
        pad_right = WS_SMALL
        self.logger.debug(f" Shape of signal before padding : {self.x.shape}")
        self.x_padded = np.pad(self.x, pad_width=((pad_left, pad_right)))
        self.logger.debug(
            f" Shape of signal_padded after padding : {self.x_padded.shape}"
        )

    def get_numOfCols(self):
        """Function to calculate the number of columns / frames in the spectrogram given the signal length, window and hop sizes

        Parameters:
        None

        Returns:
        number of columns in the spectrogram
        """
        self.pad_signal()
        numOfCols = np.floor((len(self.x_padded) - WS_SMALL) / HS_SMALL) + 1
        return numOfCols

    def get_STFT_smallWindow(self):
        """Function to compute STFT of the signal given the small window and hop sizes

        Parameters:
        None

        Returns:
        Matrix containing the STFT of the signal
        """
        numOfCols = self.get_numOfCols()
        X = np.zeros((WS_SMALL, int(numOfCols)))
        for col in range(int(numOfCols)):
            start_index = col * int(HS_SMALL)
            end_index = start_index + WS_SMALL
            X[:, col] = np.multiply(
                self.x_padded[start_index:end_index], self.smallWindow
            )

        X = fft(X, axis=0)
        return X

    def overlap_add(self, x_prev, x_curr, x_next):
        """Function to perform overlap-addition of 3 consecutive frames

        Parameters:
        x_prev : time domain signal corresponding to previous frame
        x_curr : time domain signal corresponding to current frame
        x_next : time domain signal corresponding to next frame

        Returns:
        combined signal obtained after overlap-addition
        """
        x1 = np.zeros(
            WS_LARGE,
        )
        x2 = np.zeros(
            WS_LARGE,
        )
        x3 = np.zeros(
            WS_LARGE,
        )
        x4 = np.zeros(
            WS_LARGE,
        )

        x1[:HS_SMALL] = x_prev[:HS_SMALL]
        x2[HS_SMALL : 2 * HS_SMALL] = x_prev[HS_SMALL:] + x_curr[:HS_SMALL]
        x3[2 * HS_SMALL : 3 * HS_SMALL] = x_curr[HS_SMALL:] + x_next[:HS_SMALL]
        x4[3 * HS_SMALL :] = x_next[HS_SMALL:]
        x_combined = x1 + x2 + x3 + x4
        return x_combined

    def reconstruct_time_domain(self, adaptive_spectrogram, windowType):
        """Function to reconstruct the time domain signal from the adaptive spectrogram

        Parameters:
        adaptive_spectrogram : matrix representing the adaptive spectrogram
        windowType : for each column in the adaptive spectrogram, windowType indicates whether the column was formed due to overlap addition or zero padding

        Returns:
        reconstructed signal
        """
        numOfCols = adaptive_spectrogram.shape[1]
        reconstructed_signal_temp = ifft(adaptive_spectrogram, axis=0)
        reconstructed = np.zeros(self.x_padded.shape)
        curr_length = int(HS_SMALL)

        for col in range(numOfCols):
            if windowType[col] == 1:
                startIndex = curr_length - int(HS_SMALL)
                endIndex = startIndex + WS_SMALL - 1
                reconstructed[startIndex : endIndex + 1] = (
                    reconstructed[startIndex : endIndex + 1]
                    + reconstructed_signal_temp[
                        HS_SMALL : HS_SMALL + WS_SMALL - 1 + 1, col
                    ]
                )
                curr_length = endIndex + 1

            else:
                startIndex = curr_length - int(HS_SMALL)
                endIndex = startIndex + WS_LARGE - 1
                reconstructed[startIndex : endIndex + 1] = (
                    reconstructed[startIndex : endIndex + 1]
                    + reconstructed_signal_temp[:, col]
                )
                curr_length = endIndex + 1

        if not np.allclose(self.x_padded, reconstructed, atol=TOL):
            self.logger.info(
                f"The original and reconstructed are not within tolerance of {TOL}"
            )
        return reconstructed

    def generate_adaptive_spectrogram(self):
        """Function to construct the adaptive spectrogram
        Parameters:
        None

        Returns:
        adaptive spectrogram and the reconstructed signal
        """
        X = self.get_STFT_smallWindow()
        X = X[: int(np.round(WS_SMALL / 2)) + 1, :]

        # Adding 2 random columns at the end
        X = np.c_[X, np.random.randn(int(WS_SMALL / 2) + 1, 2)]
        absX = np.abs(X)
        self.logger.debug(
            f" Shape of matrix after taking the absolute value : {X.shape}"
        )

        curr = 1
        size_spectrogram = absX.shape
        adaptiveSpectrogram = []
        windowType = []

        while curr < size_spectrogram[1] - 1:
            prev = curr - 1
            next = curr + 1

            X_prev_conj = np.conj(X[-2:0:-1, prev])
            X_curr_conj = np.conj(X[-2:0:-1, curr])
            X_next_conj = np.conj(X[-2:0:-1, next])

            X_prev = np.r_[X[:, prev], X_prev_conj]
            X_curr = np.r_[X[:, curr], X_curr_conj]
            X_next = np.r_[X[:, next], X_next_conj]

            x_prev = np.multiply(ifft(X_prev, axis=0), self.smallWindow)
            x_curr = np.multiply(ifft(X_curr, axis=0), self.smallWindow)
            x_next = np.multiply(ifft(X_next, axis=0), self.smallWindow)

            corr_prev, _ = pearsonr(x=absX[:, prev], y=absX[:, curr])
            corr_next, _ = pearsonr(x=absX[:, curr], y=absX[:, next])

            if (corr_prev >= self.corr_threshold) & (
                corr_next >= self.corr_threshold
            ):  # Combined the three
                curr += 3
                x_combined = self.overlap_add(x_prev, x_curr, x_next)
                X_combined = fft(x_combined)
                adaptiveSpectrogram.append(X_combined)
                windowType.append(0)

            else:  # Zero Padding
                pad_left = int(HS_SMALL)
                pad_right = int(HS_SMALL)
                x_prev = np.pad(x_prev, pad_width=((pad_left, pad_right)))
                X_prev = fft(x_prev)
                adaptiveSpectrogram.append(X_prev)
                windowType.append(1)
                curr += 1

        adaptiveSpectrogram = np.array(adaptiveSpectrogram).T
        adaptiveSpectrogram_half = adaptiveSpectrogram[: int(WS_LARGE / 2) + 1, :]
        x_recon = self.reconstruct_time_domain(adaptiveSpectrogram, windowType)

        return adaptiveSpectrogram, x_recon
