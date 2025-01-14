# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import logging
# from ml_pipeline.config import Config

# # Set up logging
# logging.basicConfig(level=Config.LOG_LEVEL, filename=Config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# def save_plot(fig, filename):
#     """Save the plot to the specified directory."""
#     os.makedirs(Config.PLOTS_PATH, exist_ok=True)
#     plot_path = os.path.join(Config.PLOTS_PATH, filename)
#     fig.savefig(plot_path)
#     plt.close(fig)
#     logger.info(f"Plot saved to {plot_path}")

# def perform_eda(df):
#     """Perform exploratory data analysis and save plots."""
#     logger.info("Starting EDA")
    
#     # Display basic information
#     logger.info("Data Head:\n%s", df.head())
#     logger.info("Data Shape: %s", df.shape)
#     logger.info("Null Values:\n%s", df.isnull().sum())
    
#     # Plot distributions
#     logger.info("Generating pairplot")
#     pairplot = sns.pairplot(df)
#     save_plot(pairplot.fig, "pairplot.png")
    
#     # # Correlation matrix
#     # logger.info("Generating correlation heatmap")
#     # corr = df.corr()
#     # plt.figure(figsize=(10, 8))
#     # heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm')
#     # save_plot(heatmap.get_figure(), "correlation_heatmap.png")
    
#     logger.info("EDA completed")

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from ml_pipeline.config import Config

# Set up logging
logging.basicConfig(level=Config.LOG_LEVEL, filename=Config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def save_plot(fig, filename):
    """Save the plot to the specified directory."""
    os.makedirs(Config.PLOTS_PATH, exist_ok=True)
    plot_path = os.path.join(Config.PLOTS_PATH, filename)
    fig.savefig(plot_path)
    plt.close(fig)
    logger.info(f"Plot saved to {plot_path}")

def perform_eda(df):
    """Perform exploratory data analysis and save plots."""
    logger.info("Starting EDA")
    
    # Check if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame for EDA.")
    
    # Display basic information
    logger.info("Data Head:\n%s", df.head())
    logger.info("Data Shape: %s", df.shape)
    logger.info("Null Values:\n%s", df.isnull().sum())
    
    # Plot distributions
    logger.info("Generating pairplot")
    pairplot = sns.pairplot(df)
    save_plot(pairplot.fig, "pairplot.png")
    
    # Correlation matrix
    logger.info("Generating correlation heatmap")
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm')
    save_plot(heatmap.get_figure(), "correlation_heatmap.png")
    
    logger.info("EDA completed")