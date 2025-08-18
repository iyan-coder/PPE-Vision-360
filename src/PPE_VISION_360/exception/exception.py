import sys

from src.PPE_VISION_360.logger.logger import logger


class PpeVision360Exception(Exception):
    """
    Custom exception class for ML systems related to Student Performance.

    Provides detailed error reporting with file name, line number,
    and the original error message, making debugging easier.
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        """
        Constructor for StudentPerformance.

        Args:
            error_message (Exception): The original exception object.
            error_detail (sys): Pass the sys module for traceback extraction.
        """
        super().__init__(str(error_message))  # Base class init with basic error message
        self.error_message = self.get_detailed_error_message(
            error_message, error_detail
        )

    def get_detailed_error_message(self, error: Exception, error_detail: sys) -> str:
        """
        Builds a detailed error message including file name and line number.

        Args:
            error (Exception): The actual raised error object.
            error_detail (sys): The sys module to access exception info.

        Returns:
            str: A formatted string describing where and what went wrong.
        """
        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"[ERROR] {error} | File: {file_name}, Line: {line_number}"
        else:
            return f"[ERROR] {error} (No traceback info available)"

    def __str__(self) -> str:
        """
        Returns the detailed error message when the exception is printed.
        """
        return self.error_message


# if __name__ == "__main__":
#     try:
#         logger.info("Enter the try block")
#         x = 1 / 0  # Force an error
#     except Exception as e:
#         logger.error("Exception occurred", exc_info=True)
#         raise StudentPerformanceException(e, sys)