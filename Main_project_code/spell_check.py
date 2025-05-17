import json

queries_json = json.load(open("cranfield/tokenized_queries.txt", 'r'))
vocabulary_queries = list(set([word for doc in queries_json for sentence in doc for word in sentence]))

def damerau_levenshtein(s1, s2):
    m = len(s1)
    n = len(s2)
    d = [[0]*(n+1) for _ in range(m+1)]
    
    for i in range(m+1):
        d[i][0] = i
    for j in range(n+1):
        d[0][j] = j
        
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,      # Deletion
                d[i][j-1] + 1,      # Insertion
                d[i-1][j-1] + cost   # Substitution
            )
            # Transposition check (Damerau extension)
            if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                d[i][j] = min(d[i][j], d[i-2][j-2] + 1)
    return d[-1][-1]

def spell_check(word, dictionary, max_distance=2):
    suggestions = []
    for correct_word in dictionary:
        if abs(len(correct_word) - len(word)) > max_distance:
            continue                                                        # Skipping obviously mismatched lengths
        distance = damerau_levenshtein(word.lower(), correct_word.lower())
        if distance <= max_distance:
            suggestions.append((correct_word, distance))
    return sorted(suggestions, key=lambda x: (x[1], -len(x[0])))[:5]

