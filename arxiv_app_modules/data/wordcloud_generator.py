# Import necessary libraries
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud_from_csv(csv_file):
    # Read the csv file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Check if the 'summary' column exists
    if 'Summary' not in df.columns:
        raise ValueError("The CSV file does not contain a 'summary' column.")
    
    # Concatenate all summaries into a single string
    text = ' '.join(df['Summary'].dropna())
    
    # Generate a word cloud image
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = None, 
                min_font_size = 10).generate(text)
    
    # Display the generated image:
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    
    # Save the word cloud image to a file
    wordcloud.to_file("wordcloud_output.png")
    
    print("Word cloud generated and saved as 'wordcloud_output.png'.")

# Example usage
if __name__ == "__main__":
    csv_file = "example_output.csv"
    generate_wordcloud_from_csv(csv_file)