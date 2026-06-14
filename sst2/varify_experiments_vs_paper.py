import csv
with open("/data/youyaru/CusText-main/CusText/sst2/experiment_results/results_statistics.csv", "r") as f:
    reader = csv.reader(f)
    base_reader = csv.reader(open("/data/youyaru/CusText-main/CusText/sst2/experiment_results/baseline_results_statistics.csv", "r"))
    next(reader)
    next(base_reader)
    counter = 0

    base_acc = next(base_reader)[0]
    print(base_acc)

    sub_accs = []
    for row in reader:
        counter += 1
        print(float(row[1])-float(base_acc))
        sub_accs.append(float(row[1])-float(base_acc))
        if counter == 3:
            break
        
paper_accs = [0.6985,0.7172,0.7029]
paper_base_acc = 0.9050
sub_accs_in_paper = []
for i in range(len(paper_accs)):
    sub_accs_in_paper.append(paper_accs[i]-paper_base_acc)
print("=======paper里面的数据和paper里面的baseline的数据之差=========")
print(sub_accs_in_paper)
print("=======sub_accs===========")
print(sub_accs)