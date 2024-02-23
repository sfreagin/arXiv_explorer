# Import necessary libraries
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_custom_colormap():
    # Define the colors for the custom colormap
    colors = ['black', 'firebrick', 'blue']  # black, red, blue

    # Define the boundaries for each color
    bounds = [0, 0.2, 0.8, 1]

    # Create a colormap object using ListedColormap
    cmap = mcolors.ListedColormap(colors)

    return cmap, bounds

def generate_wordcloud_from_csv(csv_file):
    # Read the csv file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Check if the 'summary' column exists
    if 'Summary' not in df.columns:
        raise ValueError("The CSV file does not contain a 'summary' column.")
    
    # Concatenate all summaries into a single string
    text = ' '.join(df['Summary'].dropna())
    
    # Create the custom colormap
    custom_cmap, bounds = create_custom_colormap()
    
    # Generate a word cloud image
    wordcloud = WordCloud(width=800, height=800, 
                          background_color='white', 
                          colormap=custom_cmap,
                          contour_color='black',
                          contour_width=2,
                          min_font_size=10).generate(text)
    
    # Display the generated image:
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    # Save the word cloud image to a file
    wordcloud.to_file("wordcloud_output.png")
    
    print("Word cloud generated and saved as 'wordcloud_output.png'.")

# Example usage
if __name__ == "__main__":
    csv_file = "example_output.csv"
    generate_wordcloud_from_csv(csv_file)