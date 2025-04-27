from util import *

class FF(AbstractRNNCell):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.j_h = tf.keras.layers.Dense(units)
        self.j_x = tf.keras.layers.Dense(units)
        self.k_h = tf.keras.layers.Dense(units)
        self.k_x = tf.keras.layers.Dense(units)

    @property
    def state_size(self):
        return self.units  # Single integer for single state

    def call(self, inputs, states):
        prev_output = states[0]
        j = tf.sigmoid(self.j_x(inputs) + self.j_h(prev_output))
        k = tf.sigmoid(self.k_x(inputs) + self.k_h(prev_output))
        output = j * (1 - prev_output) + (1 - k) * prev_output
        return output, [output]  # Return output and new state

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class QueryCompleter:
    def __init__(self, model_path="query_completer.keras"):
        self.model = load_model(model_path, custom_objects={'FF': FF})
        self.tokenizer = Tokenizer()
        self._load_artifacts()
        
    def _load_artifacts(self):
        with open("cranfield/cran_queries.json", 'r') as f:
            queries_json = json.load(f)
        queries = [q["query"] for q in queries_json]
        self.tokenizer.fit_on_texts(' '.join(queries).split('.'))
        self.reverse_word_index = {v: k for k, v in self.tokenizer.word_index.items()}

    def complete(self, incomplete_query, next_n_words=6, max_seq_len=50):
        seed_text = incomplete_query
        for _ in range(next_n_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
            predicted_index = np.argmax(self.model.predict(token_list, verbose=0))
            seed_text += " " + self.reverse_word_index.get(predicted_index, '')
        return seed_text
