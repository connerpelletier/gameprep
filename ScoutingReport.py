import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import random
import pandas as pd



def determine_category(row):
    BA = row['ba']
    SLG = row['slg']
    whiffp = row['whiffs']/row['swings']

    if SLG >= .420 and whiffp >= .24:
        return 'Power'
    elif BA > .250 and whiffp < .22:
        return 'Contact'
    elif BA < .200 and whiffp > .27 and SLG <= .400:
        return 'Bad'
    else:
        return 'Balanced'



#Get input name & side
input_name = 'Andrus, Elvis'
input_side = 'L'

#read data
batters = pd.read_csv("Batters Master.csv")
pitches = pd.read_csv("Kershaw Pitches.csv")

#only batters with large enough sample size
batters = batters[batters['abs'] > 100]

#put each batter into category
batters['Category'] = batters.apply(determine_category, axis=1)

#only include pitches to correct side
pitches = pitches[pitches['stand'] == input_side]

#finds category of input_name
input_category = batters.loc[batters['player_name'] == input_name, 'Category'].values[0]

#all hitter ids in same category
matching_ids = batters.loc[batters['Category'] == input_category, 'player_id'].to_list()

#final dataset, includes only pitches to corresponding side and category
pitch_data = pitches[pitches['batter'].isin(matching_ids)]



#plotting pies
fig, ax = plt.subplots(4, 3)
for b in range(0,4):
    for s in range(0,3):
        data = pitch_data.loc[(pitch_data['balls'] == b) & (pitch_data['strikes'] == s), 'pitch_type'].value_counts().to_dict()
        
        wedges, texts = ax[b,s].pie(data.values(), labels=data.keys(), startangle=90)
        for i, wedge in enumerate(wedges):
            wedge.set_picker(5) # Tolerance in points
            
        ax[b,s].set_label("{}-{}".format(b,s))
        if ((s == 0) & (b == 0)):
            ax[b, s].set_ylabel("Ball {}".format(b))
            ax[b, s].set_title("Strike {}".format(s))
        elif (b == 0):
            ax[b, s].set_title("Strike {}".format(s))
        elif (s == 0):
            ax[b, s].set_ylabel("Ball {}".format(b))



# Define callback function to print slice label  
def on_click(event):
    if isinstance(event.artist, Wedge):
        wedge = event.artist
        pitch = wedge.get_label()
        print(pitch)
        b, s = event.mouseevent.inaxes.get_label().split("-")
        print("{}-{}".format(b,s))
        
        #tempdata = pitch_data.loc[(pitch_data['balls'] == b) & (pitch_data['strikes'] == s) & (pitch_data['pitch_type'] == pitch), :]
        #speed = np.mean(tempdata['effective_speed'])
        
        
        speed = np.mean(pitch_data.loc[(pitch_data['balls'] == b) & (pitch_data['strikes'] == s) & (pitch_data['pitch_type'] == pitch), 'effective_speed'])
        
        #print("{}".format(tempdata.iloc[1,5]))
        print('effective pitch speed: {}'.format(speed))
        

        
                  

fig.canvas.mpl_connect('pick_event', on_click)    

# Show plot
plt.show()