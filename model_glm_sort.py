from sklearn.linear_model import SGDClassifier
from itertools import islice
from sklearn.metrics import accuracy_score
import xgboost as xgb
from openai import AzureOpenAI
import pandas as pd
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import ast
from collections import OrderedDict
import causalml as cm
import copy
import bnlearn as bn
import numpy as np
from collections import defaultdict, deque
import time

def extract_json_dag(text: str) -> dict:
    """
    Extract the *first* balanced {...} block and return it as a Python dict.
    Works with both real JSON and Python‑style single‑quoted dicts.
    """
    tag = re.search(r"<\s*Causal\s+structure\s*>(.*?)</\s*Causal\s+structure\s*>",
                    text, re.IGNORECASE | re.DOTALL)
    candidate = tag.group(1) if tag else text

    start = candidate.find('{')
    if start == -1:
        raise ValueError("No opening brace '{' found")

    depth = 0
    for i, ch in enumerate(candidate[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                block = candidate[start:i + 1]
                break
    else:
        raise ValueError("Unbalanced braces – no matching '}' found")

    try:
        return json.loads(block)
    except json.JSONDecodeError:
        pass  # fall through to Python literal

    try:
        return ast.literal_eval(block)
    except Exception as e:
        raise ValueError(f"Unable to parse object: {e}")
    
def extract_json(input_string):
    start_pos = input_string.find('[')  # Find the start of the JSON object
    if start_pos == -1:
        return None  # No JSON object found

    # Track the nesting level of the JSON string
    nesting_level = 0
    for i in range(start_pos, len(input_string)):
        if input_string[i] == '[':
            nesting_level += 1
        elif input_string[i] == ']':
            nesting_level -= 1
        
        if nesting_level == 0:  # End of the JSON string
            end_pos = i + 1  # Include the closing '}'
            json_string = input_string[start_pos:end_pos]
            json_string.replace('\n','')
            #print(json_string)
            #data_ = ast.literal_eval(json_string)
            try: 
                data_ = ast.literal_eval(json_string)
                return data_
            except:
                continue
    return None  # JSON string did not end properly

def get_causal_graph(df: pd.DataFrame, methodtype='hc', scoretype='bic'):
    causal_model = bn.structure_learning.fit(df, methodtype, scoretype)
    return causal_model['model_edges']

def build_causal_graph(edges, cols, var_types=None):
    """
    Build a JSON-style causal graph from edge list and variable names.

    Args:
        edges (list of tuple): Each tuple is (parent, child)
        cols (list of str): All variable names
        var_types (dict, optional): Mapping from variable name to its type

    Returns:
        dict: Causal graph in JSON style
    """
    # Initialize parent list for each variable
    parents_map = defaultdict(list)
    for parent, child in edges:
        parents_map[child].append(parent)

    # Build final dictionary
    causal_graph = {}
    for var in cols:
        causal_graph[var] = {
            "parents": parents_map.get(var, [])
        }

    return causal_graph

def topological_sort(causal_graph):
    """
    Topologically sort the variables in a causal graph so that parents appear before children.
    
    Args:
        causal_graph (dict): The parsed causal graph dictionary.
    
    Returns:
        list: Topologically sorted list of variable names.
    """
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for child, meta in causal_graph.items():
        for parent in meta.get("parents", []):
            graph[parent].append(child)
            in_degree[child] += 1
        if child not in in_degree:
            in_degree[child] = 0  # ensure every node appears at least once

    queue = deque([node for node in causal_graph if in_degree[node] == 0])
    sorted_vars = []

    while queue:
        node = queue.popleft()
        sorted_vars.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_vars) != len(causal_graph):
        raise ValueError("Cycle detected in the causal graph.")

    return sorted_vars


def extract_and_sort_causal_graph(text):
    """
    Main function that extracts JSON from text and returns sorted variable list.
    """
    causal_graph = extract_json_dag(text)
    sorted_vars = topological_sort(causal_graph)
    return sorted_vars

def init_tobe_refined(df, cols, num_samples, methodtype, scoretype, sort):
    causal_edges = get_causal_graph(df[cols], methodtype, scoretype)
    causal_graph = build_causal_graph(causal_edges, cols)
    prompt = f'''<Causal structure>Here is the causal structure of the data:
{causal_graph}</Causal structure>

<Task> The ultimate goal is to produce accurate and convincing synthetic
data that dutifully represents these causal relationships given the user provided samples. </Task>
'''
    sorted_var = topological_sort(causal_graph)
    if sort:
        df_sorted = df[sorted_var].copy()
    else:
        df_sorted = df.copy()
    return prompt, df_sorted


class MultiAgentGAN():
    def __init__(self, gen_client, opt_client, gen_model_nm, opt_model_nm, params, real_data, cols,y_col, num_cols, meta_data, cate_desc, data_desc, logfile, gen_temperature=0.5, opt_temperature=0.5, use_fuzzy_samples=False, fuzzy_samples_num=2, num_score_pairs=3, real_samples_num=2,  methodtype="hc", scoretype="bic", use_causal_graph=True, sort=False) -> None:
        '''
        api_key, api_version, and azure_endpoint are used to define an Azure client

        params: xgboost parameters for training a discriminator
        '''
        
        self.gen_client = gen_client
        self.opt_client = opt_client
        self.gen_model_nm = gen_model_nm
        self.opt_model_nm = opt_model_nm
        self.sort = sort

        self.params = params
        self.cols = cols
        self.y_col = y_col
        self.num_cols = num_cols
        self.logfile = logfile
        real_data.reset_index(inplace=True, drop=True)
        self.data_desc = data_desc
        self.use_causal_graph = use_causal_graph
        self.sort = sort
        
        if use_causal_graph:
            self.tobe_refined, self.real_data = init_tobe_refined(real_data, cols, real_samples_num, methodtype, scoretype, sort)
        else:
            self.tobe_refined = ""

        pred_true_df = pd.get_dummies(real_data[cols])
        self.pred_Xcols = list(pred_true_df.columns)
        self.res = {} # The dictionary is used to store the generation of each round
        self.res_df = [] # The dictionary is used to store all the converted synthetic dataframes
        self.sample = None # record the few-shot samples used in this round of generation
        self.prompt_score_dict = {}
        self.res_df = []
        self.model = None
        self.evaluation = {}
        self.dict_template = {}
        self.last_epoch_prompt_score = {}

        self.meta_data = meta_data
        self.cate_desc = cate_desc
        for col in cols:
            self.dict_template[col] = []
        
        
        temp = []
        for i in range(real_samples_num):
            temptemp = '{' + f'sample {i}' + '}'
            temp.append(temptemp)
        self.response_template = str(temp)
        self.real_samples_num = real_samples_num
        self.num_score_pairs = num_score_pairs # In the optimizer, how many examples will be provided to the optimizer
        self.use_fuzzy_samples = use_fuzzy_samples
        self.fuzzy_samples_num = fuzzy_samples_num
        self.gen_temperature = gen_temperature
        self.opt_temperature = opt_temperature
        self.generator_completion_tokens = 0
        self.generator_prompt_tokens = 0
        self.optimizer_completion_tokens = 0
        self.optimizer_prompt_tokens = 0
        self.loss_record = []
        self.sampled_rows_hist = []
        self.prompt_optimize_instruction = '''Your updated prompt should explicitly include any modifications to the causal graph and guidance.  The aim is to lower the score. The updated prompt:
'''     
        self.opt_trajectory = [] # Used to record the optimization prompt records
        self.causal_graph_traj = [] # Used to record the causal graph evolution during optimization
    
    def row2dict(self, rows):
        rows.reset_index(inplace=True, drop=True)
        res = []
        for i in range(len(rows)):
            example_data = {}
            row = rows.iloc[i, :]
            for x in self.cols:
                if x in self.num_cols:
                    example_data[x] = round(row[x], 1)
                else:
                    example_data[x] = row[x]
            res.append(example_data)
        return str(res)
    
    def instruction(self, sample, refined_prompt, cond=None):
        prompt_sys = "You are skilled data generation model. Your task is to understand the instructions below and generate tabular data.\n"
        prompt_sys = prompt_sys + "<Data description>" + self.data_desc + "</Data description>\n\n"
        prompt_sys = prompt_sys + "<Data schema>" + str(self.meta_data) + "</Data schema>\n\n"
        prompt_sys += "Categorical variables and their available categories:\n"
        prompt_sys = prompt_sys + '<Categorical variables>' + str(self.cate_desc) + "<\Categorical variables>\n\n"
        
        if self.use_causal_graph:
            prompt_sys = prompt_sys + refined_prompt
        else:
            prompt_sys = """
<Task> The ultimate goal is to produce accurate and convincing synthetic
data given the user provided samples. </Task>"""

        prompt_user = f"""<example>Here are examples from real data: 
{sample}\n
<\example>
        """
        if cond:
            prompt_user += f'''
<Instruction>Generate {self.real_samples_num} synthetic samples with {cond}. Response should be formatted strictly as a list in JSON format, suitable for direct use in data processing scripts such as conversion to a DataFrame in Python. No additional text or numbers should precede the JSON data.</Instruction>'''
        else:
            prompt_user += f"<Instruction>Generate {self.real_samples_num} synthetic sample mimics the provided samples. DO NOT COPY the sample. The response should be formatted strictly as a list in JSON format, which is suitable for direct use in data processing scripts such as conversion to a DataFrame in Python. No additional text or numbers should precede the JSON data. <\Instruction>"
        return prompt_sys, prompt_user

    def gen(self, batch_size, i=0, epoch=0, num_folds=1, cond=None):
    
        res = []
        j = i
        if j + batch_size <= len(self.real_data):
            self.sample = self.real_data.loc[j:(j+batch_size), self.cols].copy()
        else:
            self.sample = self.real_data.loc[j:, self.cols].copy()
        # causal_graph = extract_causal_edges(self.tobe_refined)
        # self.cols = reorder_columns(causal_graph, self.real_data, self.y_col)
        
        if (self.use_fuzzy_samples) and (len(self.res_df)>0):
            df_train = self.real_data.copy()
            df_dummy_train = pd.get_dummies(df_train)
            df_dummy_train = self._check_cols(df_dummy_train)
            X_ = df_dummy_train.to_numpy()
            y_ = np.numpy([1] * len(df_train))
            dX_ = xgb.DMatrix(X_, label=y_)
            prob_scores = self.model.predict(dX_)
            df_train['prob_real'] = prob_scores
            fuzzy_samples =  df_train.sort_values(by='prob_real', ascending=True).reset_index(drop=True)
            fuzzy_samples = fuzzy_samples.loc[0:(self.fuzzy_samples_num-1), self.cols].copy()
            
        while j < i + batch_size:
            sampled_rows = self.real_data.loc[j : (j+self.real_samples_num-1), self.cols].copy()
            if (self.use_fuzzy_samples) and (len(self.res_df)>0):
                sampled_rows = pd.concat([sampled_rows, fuzzy_samples])

            sample = self.row2dict(sampled_rows)
            sys_info, user_info = self.instruction(sample, self.tobe_refined, cond=cond)
            try:
                resp_temp = self.gen_client.chat.completions.create(
                    model=self.gen_model_nm, 
                    messages=[
                        {"role": "system", "content": sys_info },
                        {"role": "user", "content": user_info}
                    ],
                    temperature=self.gen_temperature,
                    n = num_folds
                )
            except:
                time.sleep(60)
                resp_temp = self.gen_client.chat.completions.create(
                    model=self.gen_model_nm, 
                    messages=[
                        {"role": "system", "content": sys_info },
                        {"role": "user", "content": user_info}
                    ],
                    temperature=self.gen_temperature,
                    n = num_folds
                )
            #print(resp_temp)
            if num_folds > 1:
                for fold in range(num_folds):        
                    res.append(resp_temp.choices[fold].message.content)
            else:
                res.append(resp_temp.choices[0].message.content)

            j = j + self.real_samples_num
        index = str(epoch)+'-'+str(i)
        self.res[index] = res
    
    def process_response(self, resp_lst):
        '''
        After generating a batch of synthetic samples, this function is used to preprocess the data and save it in a dataframe
        '''
        res = {}
        for key, val in self.dict_template.items():
            res[key] = []
        self.json_err = 0
        self.no_group_err = 0
        self.var_key_err = 0
        self.dict_error = 0

        for x in resp_lst:
            try:
                json_temp = extract_json(x)
            except SyntaxError as e:
                self.json_err += 1
                # Handle the SyntaxError
            try:
                for sample in json_temp:
                    if set(sample.keys()) != set(res.keys()):
                        self.var_key_err += 1
                        continue
                    for x, val in sample.items():
                        res[x].append(val)
            except:
                self.dict_error += 1
        return pd.DataFrame.from_dict(res)
    
    def optimizer_agent(self):
        # This agent is used to optimize the prompt given instruction-score pairs
        optim_sys_info = '''You are a prompt optimizer. Your task is to optimize prompts for generating high-quality synthetic data. Aim to lower the scores associated with each casual structure and prompt, where a lower score reflects better quality. Here are the steps:
1. Examine the existing prompt-score pairs.
2. Consider the following questions: Are there any links you'd like to add or to be removed? Any whose direction should be reversed? ONLY consider the variables exist in JSON. Do not remove the variables from JSON.
3. Output a new valid JSON that you think would provide complete information for predicting the outcome and the improved representation of the causal relationships between the variables. Use the SAME structures and keys as the provided JSON data, e.g. {key1: {"parents": [...], ...}}'''

        inst_score = ""
        lowest_n = sorted(self.prompt_score_dict.items(), key=lambda x: x[1]['accu'])[:self.num_score_pairs]

        for entry in lowest_n:
            prompt = entry[1]['prompt']
            accu = entry[1]['accu']
            inst_score += ('"'+prompt + '"' + '\n')
            inst_score += f"Score: {round(accu * 100, 2)}%\n\n"
            
        with open(self.logfile, 'a') as f:
            f.write(inst_score + self.prompt_optimize_instruction + '\n')
        try:
            response = self.opt_client.chat.completions.create(
                model=self.opt_model_nm,
                messages = [
                    {"role": "system", "content": optim_sys_info},
                    {"role": "user", "content": inst_score + self.prompt_optimize_instruction}
                ],
                temperature = self.opt_temperature)
            
        except:
            time.sleep(20)
            response = self.opt_client.chat.completions.create(
                model=self.opt_model_nm,
                messages = [
                    {"role": "system", "content": optim_sys_info},
                    {"role": "user", "content": inst_score + self.prompt_optimize_instruction}
                ],
                temperature = self.opt_temperature)
        refined_prompt = response.choices[0].message.content
        self.opt_trajectory.append(refined_prompt)
        
        if self.sort:
            sorted_var = extract_and_sort_causal_graph(refined_prompt)
            self.real_data = self.real_data[sorted_var]
            self.causal_graph_traj.append(sorted_var)
        
        return refined_prompt
    
    def _check_cols(self, df):
        dummy_df_cols = list(df.columns)
        for x in self.pred_Xcols:
            if x in dummy_df_cols:
                continue
            else:
                df[x] = [0] * len(df)
        return df
    
    def _run(self, batch_size, epochs):
        for e in range(epochs):
            i = 0
            ii = 1
            self.real_data = self.real_data.sample(frac=1).reset_index(drop=True)
            while i < len(self.real_data):
                # Generate synthetic samples
                self.gen(batch_size, i, e)
                index = str(e) + '-' + str(i)
                df_temp = self.process_response(self.res[index])
                self.res_df.append(df_temp)

                # Prepare training the discriminator
                df_temp['real_identifier'] = [0] * len(df_temp)
                df_true = self.sample
                df_true['real_identifier'] = [1] * len(df_true)
                classes = np.array([0, 1])
                df_comb = pd.concat([df_temp, df_true])
                dummy_df = pd.get_dummies(df_comb[self.cols])
                dummy_df = self._check_cols(dummy_df)
                X = dummy_df[self.pred_Xcols].to_numpy()
                y = df_comb['real_identifier'].to_numpy()
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if not self.model:
                    self.model = SGDClassifier(loss = "log_loss", max_iter = 10, tol=None, warm_start = True)
                    self.model.partial_fit(X_train, y_train, classes = classes)
                else:
                    self.model.partial_fit(X_train, y_train, classes = classes)
                    
                y_pred = self.model.predict(X_test)
                accu = accuracy_score(y_test, y_pred)

                with open(self.logfile, 'a') as f:
                    f.write(f">>>>>>>>>>>>>>> Epoch: {e}, batch: {ii} <<<<<<<<<<<<<<<< \n")
                    f.write(f"Instruction: {self.tobe_refined}\n")
                    f.write(f"Number of samples: {len(df_temp)}\n")
                    f.write(f"Number of JSON extraction errors: {self.json_err}\n")
                    f.write(f"Number of group variable missing: {self.no_group_err}\n")
                    f.write(f"Number of dictionary error: {self.dict_error}\n")
                    f.write(f"Number of variable not matach: {self.var_key_err}\n")
                    f.write(f"Score: {accu}\n\n")

                # Update the instruction pair dictionary
                if len(df_temp) >= 0.8 * batch_size:
                    self.prompt_score_dict[index] = {}
                    self.prompt_score_dict[index]['prompt'] = self.tobe_refined
                    self.prompt_score_dict[index]['X_test'] = X_test
                    self.prompt_score_dict[index]['y_test'] = y_test
                    for index_temp, dict_temp in self.prompt_score_dict.items():
                        X_test_temp = dict_temp['X_test']
                        y_test_temp = dict_temp['y_test']
                        preds_temp = self.model.predict(X_test_temp)
                        accu_temp = accuracy_score(y_test_temp, preds_temp)
                        self.prompt_score_dict[index_temp]['accu'] = accu_temp
                    # start = max(0, len(self.prompt_score_dict) - 3)
                    # last_accu_values = [value['accu'] for value in islice(self.prompt_score_dict.values(), start, None)]
                    # self.loss_record.append(last_accu_values)
                    all_accu_values = [value['accu'] for value in self.prompt_score_dict.values()]
                    self.loss_record.append(all_accu_values)

                    if e == epochs - 1:
                        self.last_epoch_prompt_score[index] = {}
                        self.last_epoch_prompt_score[index]['prompt'] = self.tobe_refined
                        self.last_epoch_prompt_score[index]['accu'] = self.prompt_score_dict[index]['accu']


                # Optimize prompt
                self.tobe_refined = self.optimizer_agent()
                i += batch_size
                ii += 1
    
    def gen_without_optimization(self, num_folds = 1):
        '''
        It is used for generate a single dataset with a fixed prompt after optimization
        '''
        res = []

        refined_final = min(self.last_epoch_prompt_score, key=lambda k: self.last_epoch_prompt_score[k]['accu'])
        refined_final_prompt = self.last_epoch_prompt_score[refined_final]['prompt']
        j = 0

        while j < len(self.real_data):
            sampled_rows = self.real_data.loc[j : (j+self.real_samples_num-1), self.cols].copy()
            sample = self.row2dict(sampled_rows)
            
            sys_info, user_info = self.instruction(sample, refined_final_prompt)
            
            # with open(self.logfile, 'a') as f:
            #     f.write('@@@@@@@@@@@@@@@@@@@@@@Final generation prompt:')
            #     f.write(f'System prompt: {sys_info}')
            #     f.write(f'User prompt:{user_info}')

            try:

                resp_temp = self.gen_client.chat.completions.create(
                    model=self.gen_model_nm, 
                    messages=[
                        {"role": "system", "content": sys_info },
                        {"role": "user", "content": user_info}
                    ],
                    temperature=self.gen_temperature,
                    n = num_folds
                )
            except:
                time.sleep(60)
                resp_temp = self.gen_client.chat.completions.create(
                    model=self.gen_model_nm, 
                    messages=[
                        {"role": "system", "content": sys_info },
                        {"role": "user", "content": user_info}
                    ],
                    temperature=self.gen_temperature,
                    n = num_folds
                )
            if num_folds > 1:
                for fold in range(num_folds):
                    res.append(resp_temp.choices[fold].message.content)
            else:
                res.append(resp_temp.choices[0].message.content)

            j = j + self.real_samples_num
        
        res_df = self.process_response(res)
        return res_df
    
    def run_without_optimization(self, num_folds=1):
        '''
        This is for generation with a fixed prompt without optimization
        '''
        i = 0
        e = 0
        self.gen(len(self.real_data), i=i, epoch=e, num_folds=num_folds)
        index = str(e) + '-' + str(i)
        df_temp = self.process_response(self.res[index])
        return df_temp
    
    def run_with_fixed_discriminator(self, i0, e0, batch_size, epochs):
        self.loss_record2 = []
        for e in range(epochs):
            self.real_data = self.real_data.sample(frac=1).reset_index(drop=True)
            i = 0
            ii = 0
            ee = e + e0
            while i < len(self.real_data):
                # Generate synthetic samples
                self.gen(batch_size, i, ee)
                index = str(ee) + '-' + str(i)
                df_temp = self.process_response(self.res[index])
                self.res_df.append(df_temp)

                # Prepare training the discriminator
                df_temp['real_identifier'] = [0] * len(df_temp)
                df_true = self.sample
                df_true['real_identifier'] = [1] * len(df_true)
                classes = np.array([0, 1])
                df_comb = pd.concat([df_temp, df_true])
                dummy_df = pd.get_dummies(df_comb[self.cols])
                dummy_df = self._check_cols(dummy_df)
                X = dummy_df[self.pred_Xcols].to_numpy()
                y = df_comb['real_identifier'].to_numpy()
                preds = self.model.predict(X)
                accu = accuracy_score(y, preds)

                with open(self.logfile, 'a') as f:
                    f.write(f">>>>>>>>>>>>>>> Epoch: {e}, batch: {ii} <<<<<<<<<<<<<<<< \n")
                    f.write(f"Number of samples: {len(df_temp)}\n")
                    f.write(f"Number of JSON extraction errors: {self.json_err}\n")
                    f.write(f"Number of group variable missing: {self.no_group_err}\n")
                    f.write(f"Number of dictionary error: {self.dict_error}\n")
                    f.write(f"Number of variable not matach: {self.var_key_err}\n")
                    f.write(f"Score: {accu}\n\n")

                # Update the instruction pair dictionary
                if len(df_temp) >= 0.8 * batch_size:
                    self.prompt_score_dict[index] = {}
                    self.prompt_score_dict[index]['prompt'] = self.tobe_refined
                    self.prompt_score_dict[index]['accu'] = accu
                    self.loss_record2.append(accu)

                    self.last_epoch_prompt_score[index] = {}
                    self.last_epoch_prompt_score[index]['prompt'] = self.tobe_refined
                    self.last_epoch_prompt_score[index]['accu'] = self.prompt_score_dict[index]['accu']
                
                # Optimize prompt
                self.tobe_refined = self.optimizer_agent()
                i += batch_size
                ii += 1

    def conditional_sampling(self, cond, num_folds):
        
        res = []
        refined_final = min(self.last_epoch_prompt_score, key=lambda k: self.last_epoch_prompt_score[k]['accu'])
        refined_final_prompt = self.last_epoch_prompt_score[refined_final]['prompt']
        j = 0

        while j < len(self.real_data):
            sampled_rows = self.real_data.loc[j : (j+self.real_samples_num-1), self.cols].copy()
            self.sampled_rows_hist = []
            sample = self.row2dict(sampled_rows)
            
            sys_info, user_info = self.instruction(sample, refined_final_prompt, cond)
            
            with open(self.logfile, 'a') as f:
                f.write('@@@@@@@@@@@@@@@@@@@@@@Final generation prompt:')
                f.write(f'System prompt: {sys_info}')
                f.write(f'User prompt:{user_info}')

            resp_temp = self.gen_client.chat.completions.create(
                    model=self.gen_model_nm, 
                    messages=[
                        {"role": "system", "content": sys_info },
                        {"role": "user", "content": user_info}
                    ],
                    temperature=self.gen_temperature,
                    n = num_folds
            )
            if num_folds > 0:
                for fold in range(len(num_folds)):
                    res.append(resp_temp.choices[fold].message.content)
            else:
                res.append(resp_temp.choices[0].message.content)

            j = j + self.real_samples_num
        
        res_df = self.process_response(res)
        return res_df
    

    def conditional_sampling_without_opt(self, cond, num_folds, temperature):
        
        res = []
        j = 0

        while j < len(self.real_data):
            sampled_rows = self.real_data.loc[j : (j+self.real_samples_num-1), self.cols].copy()
            self.sampled_rows_hist = []
            sample = self.row2dict(sampled_rows)
            
            sys_info, user_info = self.instruction(sample, self.tobe_refined, cond)
            
            with open(self.logfile, 'a') as f:
                f.write('@@@@@@@@@@@@@@@@@@@@@@Final generation prompt:')
                f.write(f'System prompt: {sys_info}')
                f.write(f'User prompt:{user_info}')

            resp_temp = self.gen_client.chat.completions.create(
                    model=self.gen_model_nm, 
                    messages=[
                        {"role": "system", "content": sys_info },
                        {"role": "user", "content": user_info}
                    ],
                    temperature=temperature,
                    n = num_folds
            )
            print(resp_temp)
            if num_folds > 1:
                for fold in range(num_folds):
                    res.append(resp_temp.choices[fold].message.content)
            else:
                res.append(resp_temp.choices[0].message.content)

            j = j + self.real_samples_num
        
        res_df = self.process_response(res)
        return res_df