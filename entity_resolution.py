import pandas as pd
import numpy as np
import re
import string
from collections import defaultdict
import jellyfish
from recordlinkage.preprocessing import clean as rl_clean
import networkx as nx
import json
import os
import gc
from tqdm import tqdm




class EntityResolution:
    

    ###################  1  ########################  
    def __init__(self, data_path):
        """
        
        so
        -data_path : str -> stores the path to input data
        -df :dataframe -> stores the input data (entity_resolution.json)
        -processed_df ->dataframe : stores the processed data
        -clusters :dict -> stores the grouped data (clusters)
        -result_df :dataframe -> stores the final result
           
        """
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.clusters = None
        self.result_df = None
        


    ###################  2  ########################  
    def load_data(self):
        """ load_data from entity_resolution.json
        and
            coverts it to a pandas DataFrame
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.df = pd.DataFrame(data)
        print(f"Loaded {len(self.df)} company records.")
        return self.df
    

       ###################  3  ########################  
    def preprocess_data(self):
        """preprocess the data  for matching."""
        print("Preprocessing data...")
        df = self.df.copy()
        
        # replace 'None' strings with actual None
        for col in df.columns:
            df[col] = df[col].replace('None', None)
        
        # normalize company names
        df['normalized_name'] = df['company_name'].apply(self._normalize_company_name)
        
        # create name prefix for blocking strategy
        # use first 3 characters of normalized name
        df['name_prefix'] = df['normalized_name'].apply(
            lambda x: x[:3].lower() if x and len(x) >= 3 else '_' if x else '_'
        )
        
        # extract domain from website URL if domain is None but URL exists
        mask = (df['website_domain'].isna()) & (~df['website_url'].isna())
        df.loc[mask, 'website_domain'] = df.loc[mask, 'website_url'].apply(self._extract_domain)
        
        # normalize phone numbers
        df['normalized_phone'] = df['primary_phone'].apply(self._normalize_phone)
        
        # process commercial names
        df['commercial_names_list'] = df['company_commercial_names'].apply(
            lambda x: self._split_commercial_names(x) if x is not None else []
        )
        
        # normalize address
        df['normalized_address'] = df['main_address_raw_text'].apply(
            self._normalize_address
        )
        
        # convert coordinates to float
        for col in ['main_latitude', 'main_longitude']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # create industry fingerprint (combine industry codes and labels)
        industry_cols = [
            'naics_2022_primary_code', 'naics_vertical', 
            'main_business_category', 'main_industry', 'main_sector'
        ]
        
        df['industry_fingerprint'] = df.apply(
            lambda row: ' '.join([str(row[col]) for col in industry_cols 
                                 if row[col] is not None and str(row[col]) != 'None']),
            axis=1
        )
        
        self.processed_df = df
        print("Preprocessing complete.")
        return self.processed_df
    




       ###################  4  ########################  
    def _normalize_company_name(self, name):
        
        if name is None:
            return ''
        
      
        name = name.lower()
        
        # remove legal forms
        legal_forms = [
            'llc', 'ltd', 'inc', 'corp', 'corporation', 'gmbh', 'limited',
            'co', 'company', 'group', 'holdings', 'international', 'enterprises',
            'services', 'solutions', 'technologies', 'technology', 'systems',
            'sa', 'spa', 's.p.a', 's.p.a.', 'ag', 'ab', 'plc', 'pty', 'pty ltd',
            'srl', 's.r.l', 's.r.l.', 'bv', 'b.v', 'b.v.', 'nv', 'n.v', 'n.v.',
            'oy', 'a/s', 'as', 'lp', 'l.p', 'l.p.'
        ]
        
        for form in legal_forms:
            pattern = r'\b' + re.escape(form) + r'\b\.?'
            name = re.sub(pattern, '', name)
        
        # replace special characters with ' '
        name = re.sub(r'[^\w\s]', ' ', name)
        
        # replace multiple spaces with a single space
        name = re.sub(r'\s+', ' ', name)
        
        # remove standalone digits
        name = re.sub(r'\b\d+\b', '', name)
        
        return name.strip()
    


       ###################  5  ########################  
    def _extract_domain(self, url):
        #extract domain from URL
        if url is None:
            return None
        
        try:
            # remove https/http /www
            domain = re.sub(r'^https?://(www\.)?', '', url)
            # remove everything after the first slash
            domain = domain.split('/')[0]
            return domain
        except:
            return None
    



       ###################  6  ########################  
    def _normalize_phone(self, phone):
        # remove non-digit characters
        if phone is None:
            return None
        return re.sub(r'\D', '', phone)
    


       ###################  7 ########################  
    def _split_commercial_names(self, names):
        #split commercial names string into a list
        if names is None:
            return []
        
        # split by pipe / vertical bar
        if '|' in names:
            return [name.strip() for name in names.split('|')]
        
        return [names.strip()]
    



       ###################  8  ########################  
    def _normalize_address(self, address):
        # by removing punctuation and lowercasing
        if address is None:
            return ''
        address = address.lower()
        address = address.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(address.split())
    



       ###################  9  ########################  
    def _calculate_name_similarity(self, name1, comm_names1, name2, comm_names2):
        #similarity (company names , commercial names)
        
    
        name_similarity = 0
        
        # compare primary names
        if name1 and name2:
            name_similarity = max(name_similarity, jellyfish.jaro_winkler_similarity(name1, name2))
        
        
        #1) check primary name against commercial names
        for comm_name in comm_names2:
            if name1 and comm_name:
                norm_comm_name = self._normalize_company_name(comm_name)
                name_similarity = max(name_similarity, 
                                     jellyfish.jaro_winkler_similarity(name1, norm_comm_name))
        
        
        #2) check commercial names against primary name
        for comm_name in comm_names1:
            if comm_name and name2:
                norm_comm_name = self._normalize_company_name(comm_name)
                name_similarity = max(name_similarity, 
                                     jellyfish.jaro_winkler_similarity(norm_comm_name, name2))
        

        # 3)check commercial names against each other
        for comm_name1 in comm_names1:
            for comm_name2 in comm_names2:
                if comm_name1 and comm_name2:
                    norm_comm_name1 = self._normalize_company_name(comm_name1)
                    norm_comm_name2 = self._normalize_company_name(comm_name2)
                    name_similarity = max(name_similarity, 
                                         jellyfish.jaro_winkler_similarity(norm_comm_name1, norm_comm_name2))
        
        return name_similarity
    



       ###################  10  ########################  
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        #calculate the Haversine distance between two points in kilometers

        # check for missing values
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return float('inf')
        

        # convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # radius of earth in kilometers
        return c * r
    



       ###################  11 ########################  
    def blocking_strategy(self):
        #create blocks of companies for comparison to reduce memory usage (similiarity matrix problem occured in the first place)
    

        print("Creating blocks for efficient processing...")
        df = self.processed_df
        
        #Startegies . Block by :
        # 1)  first 3 characters of normalized name
        name_blocks = df.groupby('name_prefix').indices
        

        # 2)  website domain 
        domain_blocks = {}
        for domain, group in df.groupby('website_domain'):
            if domain is not None and domain != '':
                domain_blocks[domain] = group.index.tolist()
        

        # 3) phone number 
        phone_blocks = {}
        for phone, group in df.groupby('normalized_phone'):
            if phone is not None and phone != '':
                phone_blocks[phone] = group.index.tolist()
        

        # 4) country and region 
        location_blocks = {}
        for country, region_group in df.groupby(['main_country_code', 'main_region']):
            if None not in country and country[0] is not None and country[0] != '':
                location_blocks[country] = region_group.index.tolist()
        
        return {
            'name_blocks': name_blocks,
            'domain_blocks': domain_blocks,
            'phone_blocks': phone_blocks,
            'location_blocks': location_blocks
        }
    



       ###################  12  ########################  
    def find_similar_pairs(self, similarity_threshold=0.7):
        #find pairs of similar companies using the blocking strategy above
        print(f"Finding similar company pairs with threshold {similarity_threshold}...")
        

        # get blocks
        blocks = self.blocking_strategy()
        df = self.processed_df
        
        # define  feature  weights
        weights = {
            'name': 0.35,            #  very   important
            'domain': 0.25,          #  strong  identifier
            'phone': 0.15,              #  can be a good identifier but   might  change
            'location': 0.15,        #  important
            'industry': 0.1,         #  can help resolve ambiguous cases
        }
        

        # dictionary to store similar pairs
        # process blocks to find similar pairs
        similar_pairs = []
        all_blocks = []
        

        #exact math -> good fit

        # add    X)..... blocks 
        # 1) name 
        for block_key, indices in blocks['name_blocks'].items():
            if len(indices) > 1:  # but i oonly consider blocks with at least 2 records
                all_blocks.append(('name', block_key, indices))
        
    
        ''' if len(indices) > 1:
                # add all pairs in this block with high similarity
        '''

        # 2) domain 
        for domain, indices in blocks['domain_blocks'].items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        similar_pairs.append((indices[i], indices[j], weights['domain']))
        
        # 3) phone  
        for phone, indices in blocks['phone_blocks'].items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        similar_pairs.append((indices[i], indices[j], weights['phone']))
        



        # process name blocks to find similar pairs
        print(f"Processing {len(all_blocks)} blocks...")
        for block_type, block_key, indices in tqdm(all_blocks):
            # skip very large blocks 
            if len(indices) > 1000:
                # for large blocks only process companies with similar lengths
                for i in range(len(indices)):
                    idx_i = indices[i]
                    name_i = df.loc[idx_i, 'normalized_name']
                    name_len_i = len(name_i) if name_i else 0
                    
                    # only compare with companies of similar name length
                    for j in range(i+1, len(indices)):
                        idx_j = indices[j]
                        name_j = df.loc[idx_j, 'normalized_name']
                        name_len_j = len(name_j) if name_j else 0
                        
                        # skip if name lengths are too different
                        if abs(name_len_i - name_len_j) > 5:
                            continue
                        
                        # calculate similarity and add if above threshold
                        similarity = self._calculate_pair_similarity(idx_i, idx_j, df, weights)
                        if similarity >= similarity_threshold:
                            similar_pairs.append((idx_i, idx_j, similarity))
            else:
                # process all pairs in smaller blocks
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        idx_i = indices[i]
                        idx_j = indices[j]
                        
                        # calculate similarity and add if above threshold
                        similarity = self._calculate_pair_similarity(idx_i, idx_j, df, weights)
                        if similarity >= similarity_threshold:
                            similar_pairs.append((idx_i, idx_j, similarity))
        
        print(f"Found {len(similar_pairs)} similar pairs.")
        return similar_pairs
    




       ###################  13  ########################  
    def _calculate_pair_similarity(self, idx_i, idx_j, df, weights):
        #calculate similarity between a pair of company records
       
        total_similarity = 0
        
        # get records
        record_i = df.loc[idx_i]
        record_j = df.loc[idx_j]
        


        # 1) name similarity
        name_i = record_i['normalized_name']
        name_j = record_j['normalized_name']
        comm_names_i = record_i['commercial_names_list']
        comm_names_j = record_j['commercial_names_list']
        
        name_similarity = self._calculate_name_similarity(name_i, comm_names_i, name_j, comm_names_j)
        total_similarity += weights['name'] * name_similarity
        



        # 2) domain similarity 
        domain_i = record_i['website_domain']
        domain_j = record_j['website_domain']
        
        domain_similarity = 0
        if domain_i and domain_j and domain_i.lower() == domain_j.lower():
            domain_similarity = 1.0
        
        total_similarity += weights['domain'] * domain_similarity
        



        # 3) phone similarity 
        phone_i = record_i['normalized_phone']
        phone_j = record_j['normalized_phone']
        
        phone_similarity = 0
        if phone_i and phone_j and phone_i == phone_j:
            phone_similarity = 1.0
        
        total_similarity += weights['phone'] * phone_similarity
        



        # 4)location similarity
        location_similarity = 0
        
        # check coordinates
        lat_i = record_i['main_latitude']
        lon_i = record_i['main_longitude']
        lat_j = record_j['main_latitude']
        lon_j = record_j['main_longitude']
        


        # If both records have coordinates
        if not (pd.isna(lat_i) or pd.isna(lon_i) or pd.isna(lat_j) or pd.isna(lon_j)):
            # calculate distance in km
            distance = self._haversine_distance(lat_i, lon_i, lat_j, lon_j)
            
            # convert distance to similarity (closer = more similar)
            if distance < 0.1:
                location_similarity = 1.0
            elif distance < 10:
                location_similarity = 1.0 - (distance / 10.0)
            else:
                location_similarity = 0.0
        


        # If coordinates not available, check address text
        if location_similarity == 0:
            addr_i = record_i['normalized_address']
            addr_j = record_j['normalized_address']
            
            if addr_i and addr_j:
                # check if addresses share significant portions
                words_i = set(addr_i.split())
                words_j = set(addr_j.split())
                
                if len(words_i) > 0 and len(words_j) > 0:
                    common_words = words_i.intersection(words_j)
                    # jaccard similarity
                    location_similarity = len(common_words) / len(words_i.union(words_j))
        
        total_similarity += weights['location'] * location_similarity
        



        # 5)iIndustry similarity
        industry_i = record_i['industry_fingerprint']
        industry_j = record_j['industry_fingerprint']
        
        industry_similarity = 0
        if industry_i and industry_j:
            # use Jaccard similarity for industry fingerprint
            words_i = set(industry_i.split())
            words_j = set(industry_j.split())
            
            if len(words_i) > 0 and len(words_j) > 0:
                common_words = words_i.intersection(words_j)
                industry_similarity = len(common_words) / len(words_i.union(words_j))
        
        total_similarity += weights['industry'] * industry_similarity
        
        return total_similarity
    




       ###################  14  ########################  
    def build_clusters_from_pairs(self, similar_pairs):
        #build clusters from similar pairs using graph-based clustering
        print("Building clusters from similar pairs...")
        


        # create a graph where nodes are company records
        G = nx.Graph()
        

        # add all nodes (company records)
        for i in range(len(self.processed_df)):
            G.add_node(i)
        

        # add edges between similar companies
        for idx_i, idx_j, similarity in similar_pairs:
            G.add_edge(idx_i, idx_j, weight=similarity)
        

        # find connected components (clusters)
        clusters = list(nx.connected_components(G))
        


        # convert to dictionary for easier processing
        cluster_dict = {}
        for i, cluster in enumerate(clusters):
            for node in cluster:
                cluster_dict[node] = i
        

        # handle singletons (nodes without connections)
        next_cluster_id = len(clusters)
        for i in range(len(self.processed_df)):
            if i not in cluster_dict:
                cluster_dict[i] = next_cluster_id
                next_cluster_id += 1
        

        # add cluster ID to processed dataframe
        self.processed_df['cluster_id'] = self.processed_df.index.map(cluster_dict)
        
        print(f"Identified {next_cluster_id} clusters")
        return self.processed_df
    



       ###################  15  ########################  
    def select_representative_records(self):
        #select the most representative record for each cluster
        print("Selecting representative records for each cluster...")
        

        df = self.processed_df
        result_df = df.copy()
        

        # count clusters
        clusters = df['cluster_id'].unique()
        print(f"Processing {len(clusters)} clusters")
        


        # for each cluster select the most complete and reliable record
        for cluster_id in clusters:
            cluster_records = df[df['cluster_id'] == cluster_id]
            
            # If cluster has only one record => it s the representative
            if len(cluster_records) == 1:
                continue
            

            # define columns to check for completeness in order of importance
            key_columns = [
                'company_name', 'website_domain', 'primary_phone', 
                'main_address_raw_text', 'main_latitude', 'main_longitude',
                'short_description', 'naics_2022_primary_code', 'main_business_category'
            ]
            

            # calculate completeness score for each record
            completeness_scores = {}
            for idx, record in cluster_records.iterrows():
                score = 0
                for i, col in enumerate(key_columns):
                    # weight earlier columns more heavily
                    weight = len(key_columns) - i
                    
                    # wheck if value exists and is not none
                    if not pd.isna(record[col]) and record[col] != 'None' and record[col] != '':
                        score += weight
                

                #  for newer records 
                if 'last_updated_at' in record and record['last_updated_at'] is not None:
                    try:
                        # extract year from timestamp
                        year = int(record['last_updated_at'].split('-')[0])
                        # add bonus for newer records (0-3 points)
                        if year >= 2020:
                            score += (year - 2020) + 1
                    except:
                        pass
                
                completeness_scores[idx] = score
            


            # sort by completeness score (descending)
            sorted_records = sorted(completeness_scores.items(), key=lambda x: x[1], reverse=True)
            

            # the first record in sorted_records is the most complete/representative
            representative_idx = sorted_records[0][0]
            

            # mark non-representative records for potential removal
            for idx in cluster_records.index:
                if idx != representative_idx:
                    result_df.at[idx, 'is_representative'] = False
                else:
                    result_df.at[idx, 'is_representative'] = True
        
        # mark singletons as representative
        singleton_mask = ~result_df.index.isin(result_df[result_df['is_representative'].notna()].index)
        result_df.loc[singleton_mask, 'is_representative'] = True
        
        self.result_df = result_df
        print("Representative record selection complete.")
        return result_df
    



       ###################  16  ########################  
    def create_output_dataset(self):
        #create the final output dataset with cluster information
        print("Creating final output dataset...")
        
        output_df = self.result_df.copy()
        

        # calculate cluster sizes
        cluster_sizes = output_df['cluster_id'].value_counts().to_dict()
        output_df['cluster_size'] = output_df['cluster_id'].map(cluster_sizes)
        

        # add duplicate IDs within each cluster
        for cluster_id in output_df['cluster_id'].unique():
            cluster_records = output_df[output_df['cluster_id'] == cluster_id]
            
            # skip singleton clusters
            if len(cluster_records) == 1:
                continue
            

            # find the representative record
            rep_record = cluster_records[cluster_records['is_representative'] == True]
            if len(rep_record) == 1:
                rep_idx = rep_record.index[0]
                
                # create list of duplicate indices
                duplicate_indices = [idx for idx in cluster_records.index if idx != rep_idx]
                duplicate_indices_str = ','.join(map(str, duplicate_indices))
                
                # add duplicate IDs to representative record
                output_df.at[rep_idx, 'duplicate_ids'] = duplicate_indices_str
        

        #now for final output
        # sort by cluster ID and representative status
        output_df = output_df.sort_values(['cluster_id', 'is_representative'], ascending=[True, False])
        
        print("Final output dataset ready.")
        return output_df
    




       ###################  17  ########################  
    def run_entity_resolution(self, similarity_threshold=0.7):
        #Run the complete entity resolution pipeline 
        self.load_data()
        self.preprocess_data()
        similar_pairs = self.find_similar_pairs(similarity_threshold)
        self.build_clusters_from_pairs(similar_pairs)
        self.select_representative_records()
        return self.create_output_dataset()
    


       ###################  18  ########################  
    def save_results(self, output_path):
        #save in new json file
        if self.result_df is None:
            print("No results to save. Run entity resolution first.")
            return
        
        # create a simplified output 
        output_columns = [
            'company_name', 'company_legal_names', 'company_commercial_names',
            'main_country_code', 'main_country', 'main_region', 'main_city',
            'main_address_raw_text', 'primary_phone', 'website_url', 'website_domain',
            'naics_vertical', 'main_business_category', 'main_industry', 'main_sector',
            'cluster_id', 'is_representative', 'duplicate_ids', 'cluster_size'
        ]
        
        # only keep columns that exist in the result
        output_columns = [col for col in output_columns if col in self.result_df.columns]
        
        #  output dataframe
        output_df = self.result_df[output_columns]
        
        # convert to json
        result_json = output_df.to_json(orient='records')
        
        # saveeeeeee
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result_json)
        
        print(f"Results saved to {output_path}")
        
        # save only unique records (representatives) if needed
        unique_df = output_df[output_df['is_representative'] == True]
        unique_output_path = output_path.replace('.json', '_unique.json')
        unique_json = unique_df.to_json(orient='records')
        with open(unique_output_path, 'w', encoding='utf-8') as f:
            f.write(unique_json)
        
        print(f"Unique records saved to {unique_output_path}")



if __name__ == "__main__":


    # Initialize entity resolution
    er = EntityResolution("entity_resolution.json")
    # run the pipeline
    result_df = er.run_entity_resolution(similarity_threshold=0.7)
    


    er.save_results("entity_resolution_results.json")
    
    #  summary statistics
    total_records = len(result_df)
    unique_companies = len(result_df[result_df['is_representative'] == True])
    duplicate_records = total_records - unique_companies
    
    print(f"\nEntity Resolution Summary:")
    print(f"Total records: {total_records}")
    print(f"Unique companies identified: {unique_companies}")
    print(f"Duplicate records: {duplicate_records}")
    print(f"Duplication rate: {(duplicate_records / total_records) * 100:.2f}%")