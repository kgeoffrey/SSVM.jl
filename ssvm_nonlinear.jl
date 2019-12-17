### SSVM with nonlinear kernel
using HigherOrderDerivatives
using Plots
using StatsBase
using CSV
using LinearAlgebra


p(x, α) = x + 1/α * log(1 + exp(-α*x))

function ssvm_nl(X::AbstractArray, Y::AbstractArray, w::AbstractArray, v::Real, γ::Number, α = 100)
    return v/2 * sum((p.(1 .- Y.*(X*w .- γ), α)).^2) + 1/2*(w'*w + γ.^2)
end




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

X_test, Y_test, X_train, Y_train = split(df, 0.10)


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
end


### kernel
me(X) = [X[i,:]'*X[i,:] for i in 1:size(X,1) ]

t = me(X_train)


function rbf(X, gamma)
    X = [X[i,:]'*X[i,:] for i in 1:size(X,1) ]
    kern = [norm(X[i] - X[j]) for i in eachindex(X), j in eachindex(X)]

    return exp.(-(1/(2*gamma)).*kern)
end


function ssvm_nl(X::AbstractArray, Y::AbstractArray, w::AbstractArray, v::Real, γ::Number, α = 10)
    return v/2 * sum((p.(1 .- Y.*(rbf(X, 1)*(Y.* w) .- γ), α)).^2) + 1/2*(w'*w + γ.^2)
end

function optimize_nl(X, Y, v, epochs, stepsize)
    w = rand(size(X,1))
    γ = 1

    loss = []
    for i in 1:epochs
        t1 = w -> ssvm_nl(X, Y, w, v, γ)
        t2 =  γ -> ssvm_nl(X, Y, w, v, γ)

        # newton method faster convergence
        w = w - stepsize .* (hessian(t1, w) \ gradient(t1, w))
        γ = γ - stepsize .* (derivative(t2, γ) / derivative(t2, γ, 2))

        append!(loss, ssvm(X, Y, w, v, γ))
    end

    return loss, w, γ
end


test = optimize_nl(X_train, Y_train, 0.1, 200, 0.1)


ssvm_nl(X_train, Y_train, rand(size(X_train, 1)), 1, 0.5)


t1 = w -> ssvm_nl(X_train, Y_train, w, 1, 0.5)


e = rand(size(X_train, 1))

gradient(t1, e)

## tests
import Base: convert,+
Dual{Float64}(x) = convert(Dual, x)
convert(::Type{Dual{<:Real}}, x::Real) = Dual(x, one(x))
+(x::Dual, y::Dual) = Dual(x.f .+ y.f, x.g .+ y.g)
