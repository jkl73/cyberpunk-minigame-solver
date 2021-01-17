import numpy as np
import cv2
from pytesseract import *

def main():
    matriximg = cv2.imread("./77mat.png")
    matrix, detboxes = parseMatrix(matriximg)
    
    seqimg = cv2.imread("./77seq.png")
    seq = parseSeq(seqimg)

    buffersize = 8

    for m in matrix:
        print(m)

    print("-----")

    for s in seq:
        print(s)

    print("-----")

    res_path = solve(matrix, seq, buffersize, detboxes)

    if res_path != None:
        # draw arrow
        result_img = draw_path(matriximg, res_path, detboxes)
        cv2.imshow('solution', result_img)



    cv2.waitKey(0)


    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # kernel = np.ones((2,2), np.uint8)
    # erosion = cv2.erode(threshold_img, kernel, iterations = 1)



    # custom_config = r'-l cyberpunk --psm 11'

    # details = pytesseract.image_to_string(erosion, config=custom_config)
    # print(details)



    # h, w, c = img.shape
    # boxes = pytesseract.image_to_boxes(erosion, config=custom_config) 
    # for b in boxes.splitlines():
    #     b = b.split(' ')
    #     print(b)
    #     img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
        
    #     org =(int(b[1]), h - int(b[2]))

    #     img = cv2.putText(img, b[0], org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 


    # strss = pytesseract.image_to_boxes(erosion, config=custom_config)
    # cv2.imshow('gg', threshold_img)

    # print(strss)

    # cv2.imshow('ff', erosion)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # 

def draw_path(matriximg, res_path, detboxes):
    # print(detboxes)

    thickness = 5
    color = (0, 255, 0)
    
    h, w, c = matriximg.shape

    for i in range(len(res_path) - 1):
        # print(res_path[i][0])
        # print(detboxes[res_path[i][0]][2 * res_path[i][1]])

        start_point = (detboxes[res_path[i][0]][2 * res_path[i][1]][1], h - detboxes[res_path[i][0]][2 * res_path[i][1]][2])
        end_point = (detboxes[res_path[i + 1][0]][2 * res_path[i + 1][1]][1], h - detboxes[res_path[i + 1][0]][2 * res_path[i + 1][1]][2])

        matriximg = cv2.arrowedLine(matriximg, start_point, end_point, color, thickness)

    return matriximg

def solve(mat, seq, buffersize, detboxes):

    # add seq and reward
    rewardmapping = {}
    # (reward: [seq1, seq2, ...])
    # hard code reward 1, 2, 3
    for i in range(len(seq)):
        # reward adjust 1 indexed 1, 2, 3
        reward = i + 1
        rewardmapping[reward] = [seq[i]]

    # reverse reward mapping
    reverse_reward = {}
    reverse_reward[0] = 1
    reverse_reward[1] = 2
    reverse_reward[2] = 3

    # only work for 3 seqs
    for i in range(len(seq)):
        for j in range(len(seq)):        
            if (i != j):
                first_combined = combine(seq[i] , seq[j], 8)

                reward = reverse_reward[i] + reverse_reward[j]

                if reward in rewardmapping.keys():
                    rewardmapping[reward] = rewardmapping[reward] + first_combined
                else:
                    rewardmapping[reward] = first_combined

                for k in range(len(seq)):
                    if (k != i and k != j):
                        for c in first_combined:
                            second_combined = combine(c, seq[k], 8)

                            second_reward = reward + reverse_reward[k]

                            if second_reward in rewardmapping.keys():
                                rewardmapping[second_reward] = rewardmapping[second_reward] + second_combined
                            else:
                                rewardmapping[second_reward] = second_combined
    
    rewards_list = rewardmapping.keys()

    x = []
    for k in sorted(rewards_list, reverse=True):
        for current_seq in rewardmapping[k]:
            visited = set()
            x = []
            if dfs(mat, current_seq, 0, visited, buffersize, True, 0, x):
                return x



def dfs(mat, seq, seqIndex, visited, buffersize, isRow, curIndex, path):
    
    if seqIndex >= len(seq):
        print(seq)
        print(path)
        return True

    if len(path) >= buffersize:
        return False

    canWildCard = False
    # check if can wild card
    if seqIndex == 0:
        if buffersize - len(seq) > curIndex:
            canWildCard = True

    if isRow:
        for j in range(len(mat[curIndex])):
            # check visited
            if (curIndex, j) in visited:
                continue

            if seq[seqIndex] == mat[curIndex][j] or seq[seqIndex] == 'XX':
                # use wild card
                if canWildCard:
                    visited.add((curIndex, j))
                    path.append((curIndex, j))
                    if dfs(mat, seq, seqIndex, visited, buffersize, not isRow, j, path):
                        return True
                    visited.remove((curIndex, j))
                    path.pop()

                # use matched
                visited.add((curIndex, j))
                path.append((curIndex, j))
                if dfs(mat, seq, seqIndex + 1, visited, buffersize, not isRow, j, path):
                    return True
                visited.remove((curIndex, j))
                path.pop()

            else:
                # use wild card
                if canWildCard:
                    visited.add((curIndex,j))
                    path.append((curIndex, j))
                    if dfs(mat, seq, seqIndex, visited, buffersize, not isRow, j, path):
                        return True
                    visited.remove((curIndex, j))
                    path.pop()

    else:
        for i in range(len(mat)):
            # check visited
            if (i, curIndex) in visited:
                continue

        

            if seq[seqIndex] == mat[i][curIndex]:
                # use wild card
                if canWildCard:
                    visited.add((i, curIndex))
                    path.append((i, curIndex))
                    if dfs(mat, seq, seqIndex, visited, buffersize, not isRow, i, path):
                        return True
                    visited.remove((i, curIndex))
                    path.pop()

                # use matched
                visited.add((i, curIndex))
                path.append((i, curIndex))
                if dfs(mat, seq, seqIndex + 1, visited, buffersize, not isRow, i, path):
                    return True
                visited.remove((i, curIndex))
                path.pop()

            else:
                # use wild card
                if canWildCard:
                    visited.add((i, curIndex))
                    path.append((i, curIndex))
                    if dfs(mat, seq, seqIndex, visited, buffersize, not isRow, i, path):
                        return True
                    visited.remove((i, curIndex))
                    path.pop()




# try to combine two seqs, can add padding 'XX' wildcard in middle
def combine(s1, s2, seqlimit):
    # print(s1[len(s1) - 1])
    # print(s2[0])

    res = []


    # contains case
    if contains_seq(s1, s2) and len(s1) <= seqlimit:
        res = [s1]
        return res


    
    cover_length = min(len(s1), len(s2)) - 1 

    # overlap case
    while cover_length >= 1 and (len(s1) + len(s2) - cover_length) <= seqlimit:

        overlapped = True

        for i in range(cover_length):
            if s1[len(s1) - (cover_length - i)] != s2[i]:
                overlapped = False

        if overlapped:
            # print(s1 + s2[cover_length:])
            res.append(s1 + s2[cover_length:])

        cover_length -= 1

    padding_size = 0


    # not overlap case (concat with padding)
    while padding_size + len(s1) + len(s2) <= seqlimit:
        x = s1

        for i in range(padding_size):
            x = x + ['XX']

        x = x + s2
        res.append(x)
        padding_size += 1
    return res


def contains_seq(s1, s2):
    
    if len(s1) < len(s2):
        return False


    for i in range(min(len(s1), len(s2))):
        if (s1[i] != s2[i]):
            return False
    return True


# input is the matrix image, square 
def parseMatrix(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    erosion_kernel_size = 1
    erosion_iter = 1
    psm_val = '11' # 11 12 sparse

    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    erosion = cv2.erode(threshold_img, kernel, iterations = erosion_iter)

    custom_config = r'-l cyberpunk --psm ' + psm_val

    h, w, c = img.shape
    boxes = pytesseract.image_to_boxes(erosion, config=custom_config) 

    boxlist = []
    for b in boxes.splitlines():
        b = b.split(' ')
        boxlist.append([b[0], int(b[1]), int(b[2]), int(b[3]), int(b[4])])

    dim = 0
    if (len(boxlist) == 98):
        dim = 7
    elif (len(boxlist) == 72):
        dim = 6
    elif (len(boxlist) == 50):
        dim = 5
    else:
        return "failed"

    # important
    boxlist.sort(key=lambda x:x[2])

    miny = 2147483647
    maxy = 0

    for b in boxlist:
        miny = min(miny, b[2])
        maxy = max(maxy, b[2])
        
        img = cv2.rectangle(img, (b[1], h - b[2]), (b[3], h - b[4]), (0, 255, 0), 2)        
        org =(b[1], h - b[2])
        img = cv2.putText(img, b[0], org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 

    stt = pytesseract.image_to_string(erosion, config=custom_config)

    gap = (maxy - miny) / (dim - 1) / 2

    matrixlist = []

    curthreshold = miny + gap

    boxindex = 0

    # important to sort the based on y
    for i in range(0, dim):
        matrixlist.append([])

        while boxindex < len(boxlist):

            if boxlist[boxindex][2] < curthreshold and boxlist[boxindex][2] > (curthreshold - (gap * 2)):
                matrixlist[i].append(boxlist[boxindex])
                boxindex+=1
            else:
                break
        matrixlist[i].sort(key=lambda x:x[1])
        curthreshold += (2 * gap)
    ret = []

    reversematrixlist = []

    for i in reversed(range(len(matrixlist))):
        row = []
        for j in range(0, len(matrixlist[i]), 2):
            row.append(matrixlist[i][j][0] + matrixlist[i][j+1][0])
        ret.append(row)
        reversematrixlist.append(matrixlist[i])

    # print(ret)
    cv2.imshow('f', img)
    cv2.imshow('err', erosion)

    return ret, reversematrixlist



# input is the sequence image
def parseSeq(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    erosion_kernel_size = 1
    erosion_iter = 1
    psm_val = '11' # 11 12 sparse

    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    erosion = cv2.erode(threshold_img, kernel, iterations = erosion_iter)

    custom_config = r'-l cyberpunk --psm ' + psm_val

    h, w, c = img.shape
    boxes = pytesseract.image_to_boxes(erosion, config=custom_config) 

    boxlist = []
    for b in boxes.splitlines():
        b = b.split(' ')
        boxlist.append([b[0], int(b[1]), int(b[2]), int(b[3]), int(b[4])])

    # important
    boxlist.sort(key=lambda x:x[2])

    miny = 2147483647
    maxy = 0

    font_height = 0

    for b in boxlist:        
        miny = min(miny, b[2])
        maxy = max(maxy, b[2])

        font_height = max(font_height, (b[4] - b[2]))

        img = cv2.rectangle(img, (b[1], h - b[2]), (b[3], h - b[4]), (0, 255, 0), 2)        
        org =(b[1], h - b[2])
        img = cv2.putText(img, b[0], org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 

    stt = pytesseract.image_to_string(erosion, config=custom_config)

    # print(miny)
    # print(maxy)

    matrixlist = []
    matrixlist.append([])
    curindex = 0

    curthreshold = miny + font_height

    boxindex = 0
    # important to sort the based on y
    while boxindex < len(boxlist):
        # print(boxlist[boxindex][2])

        if boxlist[boxindex][2] < curthreshold:
            # print(boxlist[boxindex][0] )
            matrixlist[curindex].append(boxlist[boxindex])
            boxindex+=1
        else:
            matrixlist[curindex].sort(key=lambda x:x[1])


            if boxindex < len(boxlist) - 1:
                curthreshold = boxlist[boxindex + 1][2] + font_height
                matrixlist.append([])
                curindex+=1
        
        matrixlist[curindex].sort(key=lambda x:x[1])
        
        # if boxlist[boxindex][2] < curthreshold and boxlist[boxindex][2] > (curthreshold - (gap * 2)):
        #     matrixlist[i].append(boxlist[boxindex])
        #     boxindex+=1
        # else:
        #     break

    # matrixlist[i].sort(key=lambda x:x[1])
    # curthreshold += (2 * gap)

    # print(matrixlist)

    cv2.imshow('fd', img)
    cv2.imshow('erdr', erosion)

    ret = []

    for i in reversed(range(len(matrixlist))):
        row = []
        for j in range(0, len(matrixlist[i]), 2):
            # print(matrixlist[i][j][0], end='')
            row.append(matrixlist[i][j][0] + matrixlist[i][j+1][0])
        ret.append(row)
        # print()


    return ret

if __name__ == "__main__":
    # execute only if run as a script
    main()
