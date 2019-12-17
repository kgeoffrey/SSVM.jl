### test smooth ssvm
using HigherOrderDerivatives
using Plots
using StatsBase
using CSV

# synthetic data
X = rand(100, 2)
Y = rand(range(-1, step = 2, 1), 100)
w = rand(size(X,2))
## functions
p(x, α) = x + 1/α * log(1 + exp(-α*x))

function ssvm(X::AbstractArray, Y::AbstractArray, w::AbstractArray, v::Real, γ::Number, α = 100)
    return v/2 * sum((p.(1 .- Y.*(X*w .- γ), α)).^2) + 1/2*(w'*w + γ.^2)
end

## testing
ssvm(X, Y, w, 1, 1)


function optimize(X, Y, v, epochs, stepsize)
    w = rand(size(X,2))
    γ = 1

    loss = []
    for i in 1:epochs
        t1 = w -> ssvm(X, Y, w, v, γ)
        t2 =  γ -> ssvm(X, Y, w, v, γ)

        # newton method faster convergence
        w = w - stepsize .* (hessian(t1, w) \ gradient(t1, w))
        γ = γ - stepsize .* (derivative(t2, γ) / derivative(t2, γ, 2))

        append!(loss, ssvm(X, Y, w, v, γ))
    end

    return loss, w, γ
end


test = optimize(X, Y, 0.1, 200, 0.1)

plot(test)

### get data here

df = convert(Matrix{Float64}, CSV.read("data.csv", delim = ","))

function split(df, samplesize)
    idx = sample(1:size(X,1), size(X,1))
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

X_test, Y_test, X_train, Y_train = split(df, .20)


test, we, g = optimize(X_train, Y_train, 0.1, 50, 0.1)

plot(test)

pred = X_test*we .- g

npred = [if i > 0 1 else (-1.) end for i in pred]


function predict(SVM, X_test)
    pred = X_test*we .- g
    pred = [if i > 0 (1.) else (-1.) end for i in pred]
    return pred
end

function accuracy(SVM, Y_test)
    return 100 - sum(npred - Y_test)/length(npred)
end^sf
