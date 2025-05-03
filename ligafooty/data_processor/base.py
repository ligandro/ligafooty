from abc import ABC, abstractmethod

class TrackingDataProcessor(ABC):
    @abstractmethod
    def process(self, **kwargs):
        """
        Abstract method to process tracking data.

        Each subclass must implement this method with its own parameters and logic.
        """
        raise NotImplementedError("Subclasses must implement the `process` method.")
    
    @abstractmethod
    def _clean_columns(self, df):
        """
        Abstract method to clean columns of a tracking DataFrame.
        Must be implemented by each subclass.
        """
        raise NotImplementedError("Subclasses must implement the `_clean_columns` method.")
    
    @abstractmethod
    def plot_single_frame(self,  **kwargs):
        """
        Abstract method to plot a single frame of tracking data.
        Must be implemented by each subclass.
        """
        raise NotImplementedError("Subclasses must implement the `plot_single_frame` method.")
    
    @abstractmethod
    def _player_possession(self,  **kwargs):
        """
        Abstract method to determine player possession.
        Must be implemented by each subclass.
        """
        raise NotImplementedError("Subclasses must implement the `_player_possession` method.")