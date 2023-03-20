from itertools import permutations
import pandas as pd


def gready_search(start_page: int, probabilities: dict[int, dict[int, float]], width: int = 1):
    sequence = [start_page]
    sequence_probability = 1
    remaining_pages = list(range(1, 101))
    remaining_pages.remove(start_page)
    queue = [(sequence, sequence_probability, remaining_pages)]
    all_sequences = []

    while len(queue) > 0:
        sequence, sequence_probability, remaining_pages = queue.pop(-1)

        if len(remaining_pages) == 0:
            all_sequences.append((sequence, sequence_probability))
            if len(all_sequences) % 1000000 == 0:
                print(len(all_sequences))
        else:
            i = 0
            sorted_probs = sorted(probabilities[sequence[-1]].items(), key=lambda x: -x[1])
            for _ in range(width):
                while sorted_probs[i][0] in sequence:
                    i += 1
                next_page = sorted_probs[i][0]
                next_remaining_pages = remaining_pages.copy()
                next_remaining_pages.remove(next_page)
                queue.append(
                    (
                        sequence + [next_page],
                        sequence_probability * probabilities[sequence[-1]][next_page],
                        next_remaining_pages
                    )
                )
    
    return sorted(all_sequences, key=lambda x: -x[1])[0]


def exhaustive_search(probabilities: dict[int, dict[int, float]]):
    sequences = {
        'sequence': [],
        'probability': [],
    }

    for sequence in permutations(list(range(1, 101))):
        sequence_prob = 1

        for page_number_a, page_number_b in zip(sequence, sequence[1:]):
            sequence_prob *= probabilities[page_number_a][page_number_b]

        sequences['sequence'].append(str(sequence))
        sequences['probability'].append(sequence_prob)

    sequences = pd.DataFrame(sequences).sort_values(by='probability', ascending=False, ignore_index=True)


def beam_search(start_page: int, probabilities: dict[int, dict[int, float]], width: int):
    sequences = [([start_page], 1)]

    for l in range(1, 100):
        candidates = []

        for sequence, prob in sequences:
            sorted_probs = sorted(probabilities[sequence[-1]].items(), key=lambda x: -x[1])

            i = 0
            for _ in range(min(width, 100 - l)):
                while sorted_probs[i][0] in sequence:
                    i += 1

                candidates.append(
                    (
                        sequence + [sorted_probs[i][0]],
                        prob * sorted_probs[i][1],
                    )
                )
                i += 1

        sequences = sorted(candidates, key=lambda x: -x[1])[:width]

    return sequences[0]
