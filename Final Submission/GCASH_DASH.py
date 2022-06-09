import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime_truncate import truncate
import plotly.graph_objects as go
from nltk import FreqDist

all_data = pd.read_pickle('Data/all_data.pkl')
dash_data = pd.read_pickle('Data/dash_data.pkl')


ratings1_wtp = pd.read_csv('Data/ratings1_wtp.csv').drop('Unnamed: 0', axis = 1)
ratings5_wtp = pd.read_csv('Data/ratings5_wtp.csv').drop('Unnamed: 0', axis = 1)


#slider parts
time_length = np.array((all_data['month'].max().year,all_data['month'].max().month)) - np.array((all_data['month'].min().year,all_data['month'].min().month))
tot_mo = tuple(time_length)[0]*12+tuple(time_length)[1]
first_mo = np.array((all_data['month'].min().year,all_data['month'].min().month))


def see_Hist(word, word2, score, score2, df=all_data): #df=all_data, word = word to search
  graph = go.Figure()
  if word:
    all = df[['tokens','month','score']].copy() 
    all['isin'] = all['tokens'].apply(lambda x: 1 if word in x else 0)
    
    if score:
      all = all[all['score']==score]      
    all = all[all['isin']==1].drop('isin', axis = 1).reset_index(drop=True)
    month_freq = all.groupby('month')['tokens'].count().reset_index().sort_values('month')
    if score: title = 'Distribution of Reviews with "' + word +'" in Rating: ' + str(score)
    else: title = 'Distribution of Reviews with "' + word +'"'
    month_freq = month_freq.rename(columns={'tokens':'Use Frequency'})
    graph.add_trace(go.Scatter(x=month_freq['month'],y=month_freq['Use Frequency'],name = word+'-' + str(score)))
  
  if word2:
    all = df[['tokens','month','score']].copy() 
    all['isin'] = all['tokens'].apply(lambda x: 1 if word2 in x else 0)
    if score2:
      all = all[all['score']==score2]
    all = all[all['isin']==1].drop('isin', axis = 1).reset_index(drop=True)
    month_freq = all.groupby('month')['tokens'].count().reset_index().sort_values('month')
    if score2: title = 'Distribution of Reviews with "' + word2 +'" in Rating: ' + str(score2)
    else: title = 'Distribution of Reviews with "' + word2 +'"'
    month_freq = month_freq.rename(columns={'tokens':'Use Frequency'})
    graph.add_trace(go.Scatter(x=month_freq['month'],y=month_freq['Use Frequency'], name=word2 +'-' + str(score2)))
  if word and word2:
    title = 'Distribution of reviews with "' + word +'" vs reviews with "' + word2 +'"  '
  if not word and not word2:
    return graph
  graph.update_layout(title=title,
                   xaxis_title='Month',
                   yaxis_title='Use Count')

  return graph
  

month_count = all_data['month'].value_counts().reset_index().rename(columns={'index':'month','month':'tot_count'}).sort_values(by=['month'], ascending=True).reset_index(drop=True)


def see_Hist_relative(word, word2, score, score2, df=all_data, month_count = month_count): #df=all_data, word = word to search

  graph = go.Figure()
  if word:
    all = df[['tokens','month','score']].copy() 
    all['isin'] = all['tokens'].apply(lambda x: 1 if word in x else 0)
    
    if score:
      all = all[all['score']==score]      
    all = all[all['isin']==1].drop('isin', axis = 1).reset_index(drop=True)
    month_freq = all.groupby('month')['tokens'].count().reset_index().sort_values('month')
    if score: title = 'Relative Distribution of Reviews with "' + word +'" in Rating: ' + str(score)
    else: title = 'Relative Distribution of Reviews with "' + word +'"'
    month_freq = month_freq.rename(columns={'tokens':'Use Frequency'})

    month_freq = month_freq.merge(month_count,how='right',on='month')
    month_freq['rel_amt'] = month_freq['Use Frequency']/month_freq['tot_count']
    graph.add_trace(go.Scatter(x=month_freq['month'],y=month_freq['rel_amt'],name = word+'-' + str(score)))
  
  if word2:
    all = df[['tokens','month','score']].copy() 
    all['isin'] = all['tokens'].apply(lambda x: 1 if word2 in x else 0)
    if score2:
      all = all[all['score']==score2]
    all = all[all['isin']==1].drop('isin', axis = 1).reset_index(drop=True)
    month_freq = all.groupby('month')['tokens'].count().reset_index().sort_values('month')
    if score2: title = 'Relative Distribution of Reviews with "' + word2 +'" in Rating: ' + str(score2)
    else: title = 'Relative Distribution of Reviews with "' + word2 +'"'
    month_freq = month_freq.rename(columns={'tokens':'Use Frequency'})

    month_freq = month_freq.merge(month_count,how='right',on='month')
    month_freq['rel_amt'] = month_freq['Use Frequency']/month_freq['tot_count']
    graph.add_trace(go.Scatter(x=month_freq['month'],y=month_freq['rel_amt'], name=word2 +'-' + str(score2)))
  if word and word2:
    title = 'Relative Distribution of reviews with "' + word +'" vs reviews with "' + word2 +'"  '
  if not word and not word2:
    return graph
  graph.update_layout(title=title,
                   xaxis_title='Month',
                   yaxis_title='Use Count')

  return graph
  


def filter_top_features(time1, time2, df = dash_data): #df - filterdate- postag- filter - grouby - filter - return #time1 = (yr,month)
  
  df = df.copy()
  df['month'] = df['month'].apply(lambda x: 1 if time1 <= tuple((x.year, x.month))<=time2 else 0)
  df = df[df['month']==1]
  score_content= df.groupby('score')['tokens'].agg(sum).reset_index()
  for i in range(5):
    try: 
      len(score_content.loc[i,'tokens'])<1
    except:
      print(i)
      score_content.loc[i,'tokens'] = ['']

  score_content['freqDist'] = score_content['tokens'].apply(lambda x: FreqDist(token for token in x if len(token)>1))
  score_content['freqDist'] = score_content['freqDist'].apply(lambda x: {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse =True)})

  f1 = [(x,y) for x,y in score_content['freqDist'][0].items()]
  f2 = [(x,y) for x,y in score_content['freqDist'][1].items()]
  f3 = [(x,y) for x,y in score_content['freqDist'][2].items()]
  f4 = [(x,y) for x,y in score_content['freqDist'][3].items()]
  f5 = [(x,y) for x,y in score_content['freqDist'][4].items()]
  
  tresh = 50
  f_intersection = set([x[0] for x in f1[:tresh]]).intersection(set([x[0] for x in f2[:tresh]])).\
        intersection(set([x[0] for x in f3[:tresh]])).intersection(set([x[0] for x in f4[:tresh]])).\
        intersection(set([x[0] for x in f5[:tresh]]))
  f1 = filter(f1, f_intersection)
  f2 = filter(f2, f_intersection)
  f3 = filter(f3, f_intersection)
  f4 = filter(f4, f_intersection)
  f5 = filter(f5, f_intersection)
  for i in [f1,f2,f3,f4,f5]:
    while len(i)<30:
      i.append(("",0))
  features = pd.DataFrame({'rank': [x for x in range(1,len(f1)+1)],
                         'rated_1': [x[0] for x in f1],
                         'rated_2': [x[0] for x in f2],
                         'rated_3': [x[0] for x in f3],
                         'rated_4': [x[0] for x in f4],
                         'rated_5': [x[0] for x in f5],
                         })
                         
  return features


def pie_maker(value, rating):
  time1 = tuple(first_mo + np.array((value[0]//12, value[0]%12)))
  time2 = tuple(first_mo + np.array((value[1]//12, value[1]%12)))
  if rating ==1: topics_df = ratings1_wtp.copy()
  if rating ==5: topics_df = ratings5_wtp.copy()
  topics_df['at'] = pd.to_datetime(topics_df['at']).apply(lambda x: truncate(x, 'month').date())
  topics_df['at'] = topics_df['at'].apply(lambda x: 1 if time1 <= tuple((x.year, x.month))<=time2 else 0)
  topics_df = topics_df[topics_df['at']==1]
  if rating ==1: titles = ['Customer Service','Service Reliability','Log-in/verification','Customer Dissatisfaction']
  if rating ==5: titles = ['Easy and Fast Transactions','App Usability','Services Offered','Customer Satisfaction']
  topics_df = pd.DataFrame({'Topic' : titles,
                            'Probs' : [topics_df['Topic 1'].sum(),
                                       topics_df['Topic 2'].sum(),
                                       topics_df['Topic 3'].sum(),
                                       topics_df['Topic 4'].sum()]
                           })
  pie = px.pie(topics_df,
               names = 'Topic',
               color = 'Topic',
               values = 'Probs'
              )
  pie = pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=12,
                          marker=dict(line=dict(color='#FFFFFF', width=3))) # <-pie_border
  return pie

def filter2 (word_freqlist:dict, filterout:list):
    return {word:word_freqlist[word] for word in word_freqlist if word not in filterout}


def filter (list, filterout):
    filtered = []
    count = 0
    for i in list:
        if count == 30:
            break
        if i[0] in filterout:
            continue
        else: 
            filtered.append(i)
            count +=1
    return filtered


import plotly.express as px
from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
import calendar

marks ={}
for i in range(tot_mo):
  if i%5==0:
    mark_time = list(first_mo + np.array((i//12, i%12)))
    if mark_time[1]>12:
        mark_time[0]+=1
        mark_time[1]-=12
    mark_time = list(mark_time)
    marks[i] = '%s/%s'%(mark_time[1], mark_time[0])


app = JupyterDash(__name__)

Intro = '''
# Top Words and their History

Below are the most frequently used words under each rating for the GCash reviews dataset. Click on a cell or type a word on the search bar to find out more!
You can also move around the slider to change the timeframe of the table!
'''

Copy_wordhist = '''
How often were these words used? Try out the search function to find out!  
   
(v.1.4 update: Added new relative frequency mode as recommended by sir GCash)


'''

Copy_pie = '''
Wanna know more about your data? Come check out this topic scanner that detects what are the most prominent topics found within the timeframe. 

'''
app.layout = html.Div([
              html.Div([dcc.Markdown(Intro),
                        dcc.RangeSlider(
                            id='my-range-slider',
                            min=0,
                            max=tot_mo,
                            step=1,
                            value=[0, 109],
                            marks=marks,
                        ),
                        html.Div(id='date-range-slider'),
                        html.Br(),
                        dash_table.DataTable(
                          id='table',
                          columns=[{"name": i, "id": i} for i in ['rank', 'rated_1', 'rated_2', 'rated_3', 'rated_4', 'rated_5']],
                          editable=True,
                          active_cell={'row': 4, 'column': 1, 'column_id': 'rated_2'},
                          fixed_rows={'headers': True},
                          style_table={'height': 300},
                          style_cell_conditional=[{'if': {'column_id': 'rated_1'},
                                                    'width': '18%'},
                                                  {'if': {'column_id': 'rated_2'},
                                                    'width': '18%'},
                                                  {'if': {'column_id': 'rated_3'},
                                                    'width': '18%'},
                                                  {'if': {'column_id': 'rated_4'},
                                                    'width': '18%'},
                                                  {'if': {'column_id': 'rated_5'},
                                                    'width': '18%'},
                                                ]
                                             )],
                        
                        ),
              html.H2("Word History"),
              dcc.Markdown(Copy_wordhist),
              html.Div(["Search: ",
                        dcc.Input(id='search_word', value='verify', type='text'),
                        dcc.Dropdown(id='hist_score', options=[
                                          {'label': 'All Ratings', 'value': ''},
                                          {'label': 'Rating: 1', 'value': 1},
                                          {'label': 'Rating: 2', 'value': 2},
                                          {'label': 'Rating: 3', 'value': 3},
                                          {'label': 'Rating: 4', 'value': 4},
                                          {'label': 'Rating: 5', 'value': 5}
                                      ],
                                      value='',
                                      style={'width': '45%', 'display': 'inline-block','align':'right'}
                                  )  
                        ]),
              html.Br(),
              html.Div(["Search: ",
                        dcc.Input(id='search_word2', value='verify', type='text'),
                        dcc.Dropdown(id='hist_score2', options=[
                                          {'label': 'All Ratings', 'value': ''},
                                          {'label': 'Rating: 1', 'value': 1},
                                          {'label': 'Rating: 2', 'value': 2},
                                          {'label': 'Rating: 3', 'value': 3},
                                          {'label': 'Rating: 4', 'value': 4},
                                          {'label': 'Rating: 5', 'value': 5}
                                      ],
                                      value='',
                                      style={'width': '45%', 'display': 'inline-block','align':'right'},
                                  )
                        ]),
              html.Br(),
              dcc.Checklist(id = 'relative_check',
                  options=[
                      {'label': 'Relative Frequency', 'value': '1'},
                  ],
                  value=[],
                  labelStyle={'display': 'inline-block'}
              ),
              dcc.Graph(id='graph'),
              html.Br(),
              html.H2("Topic Modelling: Latent Dirichlet Allocation"),
              dcc.Markdown(Copy_pie),
              dcc.RangeSlider(
                            id='pie-range-slider',
                            min=0,
                            max=tot_mo,
                            step=1,
                            value=[0, 109],
                            marks=marks
                        ),
              html.Div(id='pie-date-text'),
              dcc.RadioItems(id='pie_choice',
                                  options=[
                                      {'label': 'Rating: 1', 'value': 1},
                                      {'label': 'Rating: 5', 'value': 5},
                                  ],
                             value=1,
                             labelStyle={'display': 'inline-block'}
                              ), 
              dcc.Graph(id='pie'),
                      ],style={'margin-left': '5%',
                              'margin-right': '5%',
                               },)



@app.callback(
    Output('date-range-slider', 'children'),
    [Input('my-range-slider', 'value')]
    )
def update_slider(value):
    time1 = list(first_mo + np.array((value[0]//12, value[0]%12)))
    time2 = list(first_mo + np.array((value[1]//12, value[1]%12)))
    for time in [time1, time2]:
      if time[1]>12:
        time[0]+=1
        time[1]-=12
    text1 = str(calendar.month_name[time1[1]]) + " " + str(time1[0])
    text2 = str(calendar.month_name[time2[1]]) + " " + str(time2[0])
    return 'Here are the top words from %s to %s'%(text1, text2)

@app.callback(
    Output("table", "data"),
    Input('my-range-slider', 'value')
    )
def update_table(value):
    time1 = list(first_mo + np.array((value[0]//12, value[0]%12)))
    time2 = list(first_mo + np.array((value[1]//12, value[1]%12)))
    for time in [time1, time2]:
      if time[1]>12:
        time[0]+=1
        time[1]-=12
    ftable = filter_top_features(tuple(time1),tuple(time2)).to_dict('records')
    return ftable

@app.callback(
    Output('search_word', 'value'),
    [Input("table", "active_cell"),
     Input("table", "data")]
)
def update_search_value(active_cell, data):
    return str(data[active_cell['row']]['rated_' + str(active_cell['column'])])

@app.callback(
    Output('graph', 'figure'),
    [Input("search_word", "value"),
     Input("search_word2",'value'),
     Input("hist_score",'value'),
     Input("hist_score2",'value'),
     Input('relative_check', 'value')]
)
def update_figure(search_word,search_word2,hist_score,hist_score2, relative_check):
    if relative_check:
      return see_Hist_relative(search_word,search_word2,hist_score,hist_score2)
    else:
      return see_Hist(search_word,search_word2,hist_score,hist_score2)
    


@app.callback(
    Output('pie-date-text', 'children'),
    [Input('pie-range-slider', 'value')]
    )
def update_slider(value):
    time1 = list(first_mo + np.array((value[0]//12, value[0]%12)))
    time2 = list(first_mo + np.array((value[1]//12, value[1]%12)))
    for time in [time1, time2]:
      if time[1]>12:
        time[0]+=1
        time[1]-=12
    text1 = str(calendar.month_name[time1[1]]) + " " + str(time1[0])
    text2 = str(calendar.month_name[time2[1]]) + " " + str(time2[0])
    return 'Here\'s the topic distribution from %s to %s'%(text1, text2)

@app.callback(
    Output('pie', 'figure'),
    [Input('pie-range-slider', 'value'),
     Input('pie_choice', 'value')]
)
def update_figure(value, pie_choice):
    return pie_maker(value, pie_choice)


if __name__ == '__main__':
    app.run_server(debug=True)

