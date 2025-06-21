using Glob, HDF5, SimilaritySearch, JSON, CSV, DataFrames, Dates

parse_time(d) = DateTime(replace(d, " CEST"  => ""), dateformat"e dd u yyyy HH:MM:SS p")

function report_task1(D, team, path="results-task1/pubmed*/task1/*.h5", k=30)
    begins = ""
    ends = ""

    for line in eachline("log-task1.txt")
        m = match(r"^==== RUN BEGINS (.+)", line)
        if m !== nothing
            begins = m.captures[1]
            continue
        end
        m = match(r"^==== RUN ENDS (.+)", line)
        if m !== nothing
            ends = m.captures[1]
            continue
        end
    end
    
    begins, ends = parse_time(begins), parse_time(ends)
    total_time = (ends - begins).value / 1000

    gold = h5open("/home/sisap23evaluation/data2025/benchmark-eval-pubmed23.h5") do f
        f["otest/knns"][1:k, :]
    end

    for file in glob(path)
        h5open(file) do f
            A = attributes(f)
            let knns=f["knns"][], algo=A["algo"][], task=A["task"][], buildtime=A["buildtime"][], querytime=A["querytime"][], params=A["params"][]
                recall = macrorecall(gold, knns, k)
                push!(D, (; team, algo, task, k, recall, buildtime, querytime, params, begins, ends, total_time))
            end
        end
    end
end

function report_task2(D, team, path="results-task2/gooaq/task2/*.h5", k=15)
    begins = ""
    ends = ""

    for line in eachline("log-task2.txt")
        m = match(r"^==== RUN BEGINS (.+)", line)
        if m !== nothing
            begins = m.captures[1]
            continue
        end
        m = match(r"^==== RUN ENDS (.+)", line)
        if m !== nothing
            ends = m.captures[1]
            continue
        end
    end

    begins, ends = parse_time(begins), parse_time(ends)
    total_time = (ends - begins).value / 1000
    gold = h5open("/home/sisap23evaluation/data2025/benchmark-eval-gooaq.h5") do f
        f["allknn/knns"][1:k+1, :]
    end
    gold = [Set(filter(j -> i != j, c)) for (i, c) in enumerate(eachcol(gold))]
    
    for file in glob(path)
        h5open(file) do f
            A = attributes(f)
            let knns=f["knns"][], algo=A["algo"][], task=A["task"][], buildtime=A["buildtime"][], querytime=A["querytime"][], params=A["params"][]
                knns = [Set(c) for c in eachcol(knns)]
                recall = macrorecall(gold, knns)
                push!(D, (; team, algo, task, k, recall, buildtime, querytime, params, begins, ends, total_time))
            end
        end
    end

    sort!(D, :recall)
    @info D
end

D = DataFrame(; team=[], algo=[], task=[], k=[], recall=[], buildtime=[], querytime=[], params=[], begins=[], ends=[], total_time=[])
report_task1(D, "hforest")
report_task2(D, "hforest")
CSV.write("results-task12.csv", D)
