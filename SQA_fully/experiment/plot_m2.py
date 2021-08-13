import matplotlib.pyplot as plt

def convert(s):
    if s.find("ms") != -1:
        return float(s[:-2])
    else:
        return float(s[:-1]) * 1000

fig, ax = plt.subplots(1, 3)

m  = [16, 32, 64, 128, 256]
m2 = ["16", "32", "64", "128", "256", "512", "1024"]
style = ["--ro", "--bo", "--go", "--co", "--mo", "--yo", "--ko"]
time_gemm = []
time_judge = []
with open('temp') as f:
    lines = f.readlines()
    name  = lines[::4]
    act1  = lines[2::4]
    act2  = lines[3::4]
    gemm  = []
    judge = []
    for idx, (n, a, b) in enumerate(zip(name, act1, act2)):
        t = (n.split()[7]).split("-")[-1]
        if a.find("gemm") != -1:
            if a.find("activities") != -1:
                gemm.append(convert(a.split()[3]))
            else:
                gemm.append(convert(a.split()[1]))
        else:
            if a.find("activities") != -1:
                judge.append(convert(a.split()[3]))
            else:
                judge.append(convert(a.split()[1]))
        if b.find("gemm") != -1 :
            if b.find("activities") != -1:
                gemm.append(convert(b.split()[3]))
            else:
                gemm.append(convert(b.split()[1]))
        else:
            if b.find("activities") != -1:
                judge.append(convert(b.split()[3]))
            else:
                judge.append(convert(b.split()[1]))
        if (idx % len(m2)) == (len(m2)-1):
            time_gemm.append(gemm)
            time_judge.append(judge)
            gemm = []
            judge = []
print(time_gemm, time_judge)
for mm, g, s in zip(m, time_gemm, style): 
    ax[0].plot(m2, g, s, label = "M="+str(mm))
    ax[0].legend(loc="upper left")
    ax[0].set_ylim(0.0, 10000.0)
    ax[0].set_xlabel('blk_sz', fontsize=10)
    ax[0].set_ylabel('time (ms)', fontsize=10)
    ax[0].set_title('Local-field-update time')
for mm, j, s in zip(m, time_judge, style): 
    ax[1].plot(m2, j, s, label = "M="+str(mm))
    ax[1].legend(loc="upper left")
    ax[1].set_ylim(0.0, 3500.0)
    ax[1].set_xlabel('blk_sz', fontsize=10)
    #ax[1].set_ylabel('time (ms)', fontsize=10)
    ax[1].set_title('Judge-flipping time')
for mm, g, j, s in zip(m, time_gemm, time_judge, style): 
    tmp = [i+k for i, k in zip(g,j)]
    ax[2].plot(m2, tmp, s, label = "M="+str(mm))
    ax[2].legend(loc="upper left")
    ax[2].set_ylim(0.0, 15000.0)
    ax[2].set_xlabel('blk_sz', fontsize=10)
    #ax[2].set_ylabel('time (ms)', fontsize=10)
    ax[2].set_title('Total-annealing time')
   
plt.show()
