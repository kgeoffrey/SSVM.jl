# SSVM.jl
Smooth Support Vector Machine

Source: https://pdfs.semanticscholar.org/fb7e/5403c219b9a49135c21a4580608f8ef1520f.pdf
ftp://ftp.cs.wisc.edu/math-prog/tech-reports/98-14.pdf
http://www-personal.umich.edu/~mepelman/teaching/NLP/Handouts/NLPnotes12_5.pdf

### Example:

```julia
using CSV

df = convert(Matrix{Float64}, CSV.read("data.csv", delim = ","))
function splitthis(df, samplesize)
    idx = sample(1:size(df,1), size(df,1))
    l = length(idx)
    df = df[idx, :]
    s = Int(floor(l*samplesize))
    df_train = df[1:l-s, :]
    df_test = df[l-s:end, :]
    X_train = df_train[:,1:end .!= 5]
    Y_train = [if i == 0 (-1) else 1 end for i in df_train[:,5]]
    X_test = df_test[:,1:end .!= 5]
    Y_test = [if i == 0 (-1) else 1 end for i in df_test[:,5]]
    return X_test, Y_test, X_train, Y_train
end
X_test, Y_test, X_train, Y_train = splitthis(df, 0.10)

## linear Smooth Support Vector Machine
model = SSVM(X_train, Y_train)

## fitting model
fit!(model, 0.01)

## predict
prediction = predict(model, X_test)

## assess accuracy
accuracy(prediction, Y_test)
```
