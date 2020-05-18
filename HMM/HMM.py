# INF552 HW7
# Shih-Yu Lai 
# Shiuan-Chin Huang 
# Dan-Hui Wu
import math
import collections

def initialize(file):
    row = 0
    free_cell = []
    tower = []
    noisy = []
    with open(file) as f:
        f.readline()   # skip Grid-World:
        f.readline()   # skip the blank
        for i in range(10):
            line = f.readline()
            line = line.strip().split()
            for col, ele in enumerate(line):
                if ele == '1':
                    free_cell.append([int(row), int(col)])
            row += 1

        for skip in range(4):
            f.readline()   # skip words and blanks
        for tower_num in range (4):
            tower.append([int(tower_num) for tower_num in f.readline().split()[2:]])

        for skip in range(4):
            f.readline()   # skip words and blanks
        for step in range (11):
            noisy.append([float(step) for step in f.readline().split()])

    return free_cell, tower, noisy

def distance(free_cells, towers):
    distance_towers = []
    for i, free in enumerate(free_cells):
        dist = []
        for j, tower in enumerate(towers):
            euclidean_dist = math.sqrt(pow(free[0] - tower[0], 2) + pow(free[1] - tower[1], 2))
            dist.append([round(euclidean_dist * 0.7, 1), round(euclidean_dist * 1.3, 1)])
        distance_towers.append(dist)
    return distance_towers

def find_cell(free_cells, noisy, distance_towers):
    prob_state = []
    for i in range(0,len(free_cells)):
        points = free_cells[i]
        count = 0
        for j in range(0,len(noisy)):
            ele = noisy[j]
            if distance_towers[i][j][0] <= ele and ele <= distance_towers[i][j][1] :
                count += 1
        if count == len(noisy):
            prob_state.append(points)
    return prob_state

def find_neighboring_cell(location, size):
    x = location[0]
    y = location[1]
    neighboring_cell = []
    if x + 1 < size:
        neighboring_cell.append((x + 1, y))
    if y + 1 < size:
        neighboring_cell.append((x, y + 1))
    if x - 1 > 0:
        neighboring_cell.append((x - 1, y))
    if y - 1 > 0:
        neighboring_cell.append((x, y - 1))
    return neighboring_cell

def trans_prob(states, neighboring_cell):
    trans_prob_neighbour, total, trans_prob = collections.defaultdict(dict), collections.defaultdict(int), collections.defaultdict(dict)
    for num in states:
        total[num] = 0.0
        timestep = states[num]
        neighbour_cells = neighboring_cell[num]
        for step in timestep:
            step += 1
            for n in neighbour_cells:
                if n in states:
                    if step in states[n]:
                        if n not in trans_prob_neighbour[num]:
                            trans_prob_neighbour[num][n] = 0.0
                        trans_prob_neighbour[num][n] += 1.0
                        total[num] += 1.0
        for n in trans_prob_neighbour[num]:
            trans_prob[num][n] = trans_prob_neighbour[num][n] / total[num]
    return trans_prob

def viterbi(noisy, prob_state, transition):
    step = 0
    path = collections.defaultdict(dict)
    path[step] = collections.defaultdict(dict)
    for i in prob_state[step]:
        i = tuple(i)
        path[step][i] = {}
        path[step][i]['pre'] = ''
        path[step][i]['probability'] = 1.0 / len(prob_state[step])

    for time_step in range(1,len(noisy)):
        path[time_step] = collections.defaultdict(dict)
        for i in path[time_step - 1]:
            if i in transition:
                for nei in transition[i]:
                    if list(nei) in prob_state[time_step]:
                        if nei not in path[time_step]:
                            path[time_step][nei] = {}
                            path[time_step][nei]['pre'] = i
                            probability = path[time_step - 1][i]['probability'] * transition[i][nei]
                            path[time_step][nei]['probability'] = probability
                        else:
                            probability = path[time_step - 1][i]['probability'] * transition[i][nei]
                            if probability > path[time_step][nei]['probability']:
                                path[time_step][nei]['pre'] = i
                                path[time_step][nei]['probability'] = probability
    return path

def backtrack(possible_paths):
    max_prob, step = 0.0, 10
    cell = None
    path = []
    for num in possible_paths[step]:
        if max_prob < possible_paths[step][num]['probability']:
            max_prob = possible_paths[step][num]['probability']
            cell = num
    path.append(cell)
    for step in range(10,0,-1):
        parent_cell = possible_paths[step][cell]['pre']
        path.append(parent_cell)
        cell = parent_cell
    return path

def main():
    file = 'hmm-data.txt'
    free_cells, towers, noisy = initialize(file)
    distance_towers = distance(free_cells, towers)
    prob_state, states = collections.defaultdict(list), collections.defaultdict(list)
    for i in range(0, len(noisy)):
        prob_state[i] = find_cell(free_cells, noisy[i], distance_towers)
        for num in prob_state[i]:
            states[tuple(num)].append(i)
    neighboring_cell = collections.defaultdict(list)
    for num in states:
        neighboring_cell[num] = find_neighboring_cell(num, 10)
    transition = trans_prob(states, neighboring_cell)
    possible_paths = viterbi(noisy, prob_state, transition)
    path = backtrack(possible_paths)

    print("Noisy :")
    print(noisy)
    print("states : ")
    print(states)
    print("Path :")
    print(path[::-1])

if __name__ == "__main__":
    main()