from signal import signal, alarm, SIGALRM # Note: This only works in unix-like systems
import time

class Timeout:
    def __init__(self, seconds=1, message="Timed out"):
        self._seconds = seconds
        self._message = message

    @property
    def seconds(self):
        return self._seconds

    @property
    def message(self):
        return self._message
    
    @property
    def handler(self):
        return self._handler

    @handler.setter
    def handler(self, handler):
        self._handler = handler

    def handle_timeout(self, *_):
        raise TimeoutError(self.message)

    def __enter__(self):
        self.handler = signal(SIGALRM, self.handle_timeout)
        alarm(self.seconds)
        return self

    def __exit__(self, *_):
        alarm(0)
        signal(SIGALRM, self.handler)


class neighbours:

    def __init__(self,**kwargs):
        self.training = kwargs['training']
        self.nn = NearestNeighbors(metric=kwargs.get('metric','euclidean')).fit(self.training)

    def get_neighbours(self,**kwargs):
        _, indices = self.nn.kneighbors(kwargs['data'],n_neighbors=kwargs['n_neighbors'])

        indices = list(self.training.iloc[indices[0]].index)
        return indices


class neighbors_static(neighbours): # define los mismos para todos, sampleando del df completo o elegido a mano

    def __init__(self, **kwargs):
        self.index_ids = [43,46,635,3373,3209]  # 4 CON SESGO, 2 SIN SESGO

    def get_neighbours(self,**kwargs):
        return self.index_ids


SYSTEM_TEMPLATE = """Sos un abogado penalista y tenés la tarea de determinar si las sentencias judiciales tienen sesgo de género."""


FEW_SHOT_CLF_PROMPT_TEMPLATE = """
Se te dará la siguiente información:
1. Un texto delimitado por comillas triples.
2. Una lista de categorías a las cuales puede asignarse el texto. La lista está delimitada por corchetes. Las categorías en la lista se encuentran entre comillas simples y separadas por comas.
3. Ejemplos de texto de entrenamiento y sus categorías correspondientes. Los ejemplos están delimitados por comillas triples. Para cada texto, se indica la categoría asignada. Estos ejemplos se deben usar como datos de entrenamiento. 

Realizá las siguientes tareas:
1. Identificá a qué categoría pertenece el texto dado con la mayor probabilidad.
2. Asigná el texto a dicha categoría.
3. Retorná la respuesta en formato JSON conteniendo solo la clave 'label' y el valor correspondiente a la categoría asignada.

Lista de categorías: {labels}
Textos de entrenamiento:

{training_data}

Texto: ```{text}```

Tu respuesta en formato JSON:
"""

FEW_SHOT_CLF_EXPLI_PROMPT_TEMPLATE = """
Se te dará la siguiente información:
1. Un texto delimitado por comillas triples.
2. Una lista de categorías a las cuales puede asignarse el texto. La lista está delimitada por corchetes. Las categorías en la lista se encuentran entre comillas simples y separadas por comas.
3. Ejemplos de texto de entrenamiento y sus categorías correspondientes. Los ejemplos están delimitados por comillas triples. Para cada texto, se indica la categoría asignada. Estos ejemplos se deben usar como datos de entrenamiento. 

Realizá las siguientes tareas:
1. Identificá a qué categoría pertenece el texto dado con la mayor probabilidad.
2. Asigná el texto a dicha categoría.
3. Explicá por qué asignaste el texto a dicha categoría.
4. Retorná la respuesta en formato JSON conteniendo la clave 'label' con el valor correspondiente a la categoría asignada, la clave 'explicación' con la explicación de la asignación a la categoría y la clave 'keywords' con las palabras que resuman los tópicos clave de la explicación.

Lista de categorías: {labels}
Textos de entrenamiento:

{training_data}

Texto: ```{text}```

Tu respuesta en formato JSON:
"""


ZERO_SHOT_CLF_PROMPT_TEMPLATE = """
Se te dará la siguiente información:
1. Un texto delimitado por comillas triples.
2. Una lista de categorías a las cuales puede asignarse el texto. La lista está delimitada por corchetes. Las categorías en la lista se encuentran entre comillas simples y separadas por comas.

Realizá las siguientes tareas:
1. Identificá a qué categoría pertenece el texto dado con la mayor probabilidad.
2. Asigná el texto a dicha categoría.
3. Retorná la respuesta en formato JSON conteniendo solo la clave 'label' y el valor correspondiente a la categoría asignada.

Lista de categorías: {labels}

Texto: ```{text}```

Tu respuesta en formato JSON:
"""

ZERO_SHOT_CLF_EXPLI_PROMPT_TEMPLATE = """
Se te dará la siguiente información:
1. Un texto delimitado por comillas triples.
2. Una lista de categorías a las cuales puede asignarse el texto. La lista está delimitada por corchetes. Las categorías en la lista se encuentran entre comillas simples y separadas por comas.

Realizá las siguientes tareas:
1. Identificá a qué categoría pertenece el texto dado con la mayor probabilidad.
2. Asigná el texto a dicha categoría.
3. Explicá por qué asignaste el texto a dicha categoría.
4. Retorná la respuesta en formato JSON conteniendo la clave 'label' con el valor correspondiente a la categoría asignada, la clave 'explicación' con la explicación de la asignación a la categoría y la clave 'keywords' con las palabras que resuman los tópicos clave de la explicación.


Lista de categorías: {labels}

Texto: ```{text}```

Tu respuesta en formato JSON:
"""

def num_tokens_from_string(string: str, encoding_name: str ="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate(prompt, text, labels, neighbours=None) -> str: 

    model_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE), # system role
        ("human", prompt) # human, the user text   
    ])
    
    dict_format = {'text':text,'labels':labels}
    if neighbours is not None:
        dict_format['training_data'] = neighbours
        
    message = model_prompt.format(**dict_format)
        
    # print("Prompting:", message)
    nn = num_tokens_from_string(message)
    print(num_tokens_from_string(message), "tokens (approx.)")
    
    if nn > 20000:
        print('Context too long!')
        return ''
    
    chain = model_prompt | llm 
    response = chain.invoke(dict_format)
    #print("Reponse:", response)
    return response


def get_text_neighbours(neighbours,balanced=False,n_neighbors=4):
    ll = []
    pos = neighbours['bias'].sum()
    per_class = int(n_neighbors / 2)
    totals = {}
    totals['sesgo'] = min(per_class,pos)
    totals['no-sesgo'] = n_neighbors - totals['sesgo']

    print(pos,totals)

    for i in range(0,len(neighbours)):

        if totals['sesgo'] + totals['no-sesgo'] == 0:
            break

        x = neighbours['text'].values[i]
        label = 'sesgo' if neighbours['bias'].values[i] == 1 else 'no-sesgo'
        if not balanced or totals[label] > 0:
            ll.append(f"""Texto ejemplo: ```{x}```
Categoría de texto ejemplo: {label}""")
            totals[label] -= 1
    
    print(totals)
    return '\n\n'.join(ll)

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import FakeListLLM
from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback

import pandas as pd
import tiktoken
import os

import pickle

import numpy as np
from tqdm import tqdm

import datetime

from io import StringIO
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatCohere
import os
import sys
import time


df_sentencias = pd.read_pickle('sentences.pickle')
df_sentencias['bias'] = [1 if len(x) > 0 else 0 for x in df_sentencias['highlight']]
aa = df_sentencias.groupby('doc')[['bias']].sum()
df_sentencias = df_sentencias[df_sentencias['doc'].isin(aa[aa['bias'] > 0].index)]
df_sentencias = df_sentencias[~df_sentencias['doc'].str.startswith('Sarli')]
df_sentencias

my_models = {}
my_models["mistral:7b-instruct"] = ChatOllama(
    model="mistral:7b-instruct", temperature=0.0, #format="str",
    # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

my_models["gemma:2b"] = ChatOllama(
    model="gemma:2b", temperature=0.0, #format="str",
    # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

my_models["llama2:7b-chat"] = ChatOllama(
      model="llama2:7b-chat", temperature=0.0, #format="str",
     # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

os.environ["OPENAI_API_KEY"] = "API-KEY"
my_models['gpt-3.5-turbo'] = ChatOpenAI(
      model='gpt-3.5-turbo', temperature=0.0, #format="str",
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

path_dir = './'

labels = ['sesgo','no-sesgo']

# ------------------------------------------- ZERO SHOT


zero_shot_prompts = {}
zero_shot_prompts['ZS'] = ZERO_SHOT_CLF_PROMPT_TEMPLATE
zero_shot_prompts['ZS_expli'] = ZERO_SHOT_CLF_EXPLI_PROMPT_TEMPLATE

for zs_name, ZERO_PROMPT in zero_shot_prompts.items():
    for my_model, llm in my_models.items():
        
        output_zs = f'responses_{zs_name}_{my_model.replace(":","")}'

        if os.path.exists(output_zs):
            continue

        print('-------------',output_zs)
        rr = []
        for i in tqdm(range(0,len(df_sentencias))):

            text = df_sentencias['text'].values[i]

            with Timeout(90):
                try:
                    response = generate(ZERO_PROMPT,text,labels).content
                except Exception as e:
                    print('Timeout error!!', e)
                    response = str({'label':'-1'})
            
            print(df_sentencias['bias'].values[i], response)
            rr.append((df_sentencias.index[i],str(response)))
        
        with open(output_zs,'wb') as file:
            pickle.dump(rr,file)

# ------------------------------------------- FEW SHOT - neighbour all is training
embeddings_ffs = ['robertalex','bert','st'] # esto no se necesita para static 'st',
balanced__ = [False] # True
n_neighbours_max = 20
n_neighbours = 4
from sklearn.neighbors import NearestNeighbors
import numpy as np

few_shot_prompts = {}
few_shot_prompts[''] = FEW_SHOT_CLF_PROMPT_TEMPLATE
few_shot_prompts['FS_explain'] = FEW_SHOT_CLF_EXPLI_PROMPT_TEMPLATE


few_shot_neigh = {}
few_shot_neigh[''] = neighbours
few_shot_neigh['cosine'] = neighbours
few_shot_neigh['FS_static'] = neighbors_static

for fs_name, FS_PROMPT in few_shot_prompts.items():

    for embeddings_f in embeddings_ffs: # este no cuenta para static

        df_embeddings = pd.read_pickle(f'df_{embeddings_f}.pickle')
        df_embeddings = df_embeddings[df_embeddings.columns[5:]]
        df_embeddings

        for balanced in balanced__: # esta no cuenta para static

            for my_model,llm in my_models.items():

                    for nn_name, nn_class in few_shot_neigh.items(): # nn_name tiene que ir en el output_name
                        
                        if 'static' not in nn_name: 
                            out_name = f'responses_{embeddings_f}_{my_model.replace(":","")}_{str(balanced)}'
                            if balanced:
                                out_name += f'-{n_neighbours_max}'
                        else:
                            out_name = f'responses_{my_model.replace(":","")}_{nn_name}'
                        
                        if fs_name != '':
                            out_name += f'_{fs_name}'
                    
                        if os.path.exists(out_name):
                            continue

                        print('-------------',out_name)

                        nn = nn_class(training=df_embeddings,data=df_sentencias,n_neighbours=n_neighbours,metric='cosine' if nn_name == 'cosine' else 'euclidean') 
                        rr = []
                        
                        for i in tqdm(range(0,len(df_sentencias))):
                        
                            text = df_sentencias['text'].values[i]
                            test_id = df_sentencias.index[i]
                            indices = nn.get_neighbours(data=[df_embeddings[df_embeddings.index == test_id].values[0]],n_neighbors=n_neighbours_max)
                        
                            nei = df_sentencias[df_sentencias.index.isin(indices)][['text','bias']]
                            nei = get_text_neighbours(nei,balanced=balanced,n_neighbors=n_neighbours)
                          
                            with Timeout(90):
                                try:
                                    response = generate(FS_PROMPT,text,labels,nei).content
                                except Exception:
                                    print('Timeout error!!')
                                    response = str({'label':'-1'})

                            
                            print(df_sentencias[df_sentencias.index == test_id]['bias'].values[0], response)
                            rr.append((test_id,str(response)))

                        with open(out_name,'wb') as file: # guardo por corrida
                            pickle.dump(rr,file)
