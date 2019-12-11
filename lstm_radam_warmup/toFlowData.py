import json

# 把一行为一个属性的转化为, 时间序列的
with open("config.json") as f:
    config = json.load(f)
    
    file_list = config["to_data"]["to_flow"]["file_list"]
    save_file = config["to_data"]["to_flow"]["save_file"]

for i in range(len(file_list)):
    ua = []
    ub = []
    uc = []
    ia = []
    ib = []
    ic = []
    source = None
    try:
        print("-------------------- processingFile: " + file_list[i] + " --------------------")
        source = open(file_list[i], mode='r', encoding='utf-8')
        ua = source.readline().strip().split(', ')
        ua = [float(x) for x in ua]
        ub = source.readline().strip().split(', ')
        ub = [float(x) for x in ub]
        uc = source.readline().strip().split(', ')
        uc = [float(x) for x in uc]
        ia = source.readline().strip().split(', ')
        ia = [float(x) for x in ia]
        ib = source.readline().strip().split(', ')
        ib = [float(x) for x in ib]
        ic = source.readline().strip().split(', ')
        ic = [float(x) for x in ic]
    finally:
        if source is not None:
            source.close()        
    ua, ub, uc, ia, ib, ic = ua[0:800000], ub[0:800000], uc[0:800000], ia[0:800000], ib[0:800000], ic[0:800000]
    
    save = None
    try:
        save = open(save_file, mode="a", encoding='utf-8')
        for i in range(len(ua)):
            save.write(str(ua[i]) + ", " + str(ub[i]) + ", " + str(uc[i]) + ", " + str(ia[i]) + ", " + str(ib[i]) + ", " + str(ic[i]) + "\n")

    finally:
        ua.clear()
        ub.clear()
        uc.clear()
        ia.clear()
        ib.clear()
        ic.clear()

        if save is not None:
            save.close()


# for i in range(24, 67):
#     print('"/home/zk/workspace/NN_Python/paper_proj/data/data/realdata/motor_normal_solved_' + str(i) + '.txt",')






