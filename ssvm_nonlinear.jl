### SSVM with nonlinear kernel
using HigherOrderDerivatives
using Plots
using StatsBase
using CSV
using LinearAlgebra

### get data here

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

### other methods!

function predict(SVM, X_test)
    pred = X_test*we .- g
    pred = [if i > 0 (1.) else (-1.) end for i in pred]
    return pred
end

function accuracy(SVM, Y_test)
    return 100 - sum(npred - Y_test)/length(npred)
end


### kernels

function rbf(X::AbstractArray; gamma = 0.1)
    X = [X[i,:]'*X[i,:] for i in 1:size(X,1) ]
    kern = [norm(X[i] - X[j]) for i in eachindex(X), j in eachindex(X)]
    return exp.(-(1/(2*gamma)).*kern)
end

function linear(X::AbstractArray)
    return X*X'
end

### test data ###

X = rand(100, 2)
Y = rand(range(-1, step = 2, 1), 100)
w = rand(size(X,2))
new  = rbf(X)

p(x, α) = x + 1/α * log(1 + exp(-α*x))

function gssvm_loss(X::AbstractArray, Y::AbstractArray, w::AbstractArray, C::Real, γ::Number, α = 10)
    return C/2 * sum((p.(1 .- Y.*(X * Y.*w .- γ), α)).^2) + 1/2*(w'*w + γ.^2)
end

function ssvm_loss(X::AbstractArray, Y::AbstractArray, w::AbstractArray, v::Real, γ::Number, α = 100)
    return v/2 * sum((p.(1 .- Y.*(X*w .- γ), α)).^2) + 1/2*(w'*w + γ.^2)
end

t1 = w -> ssvm_nl(new, Y, w, 1, 0.5)
e = rand(size(X, 1))

@time gradient(t1, e)

function fit(X, Y, C, epochs, stepsize)
    X = kernel(X)
    w = rand(size(X,1))
    γ = 1
    loss = []
    for i in 1:epochs
        t1 = w -> ssvm_nl(X, Y, w, C, γ)
        t2 =  γ -> ssvm_nl(X, Y, w, C, γ)
        # newton method faster convergence
        w = w - stepsize .* (hessian(t1, w) \ gradient(t1, w))
        γ = γ - stepsize .* (derivative(t2, γ) / derivative(t2, γ, 2))
        append!(loss, ssvm_nl(X, Y, w, C, γ))
    end
    return loss, w, γ
end

test, ww ,yy = fit(new, Y, 0.1, 100, 0.1)

plot(test)


mutable struct GSSVM
    kernel::Function
    X::AbstractArray
    Y::AbstractArray
    w::AbstractArray
    γ::Real
    loss::AbstractArray
    function GSSVM(kernel::Function, X::AbstractArray, Y::AbstractArray)
        X = kernel(X)
        w = rand(size(X,1))
        γ = 1
        loss = []
        new(kernel, X, Y, w, γ, loss)
    end
end



model = GSSVM(rbf, X, Y)

function fit(obj::SSVM, C::Real, epochs, stepsize)
    for i in 1:epochs
        t1 = w -> ssvm_nl(obj.X, obj.Y, w, C, obj.γ)
        t2 =  γ -> ssvm_nl(obj.X, obj.Y, obj.w, C, γ)
        # newton method faster convergence
        obj.w = obj.w - stepsize .* (hessian(t1, obj.w) \ gradient(t1, obj.w))
        obj.γ = obj.γ - stepsize .* (derivative(t2, obj.γ) / derivative(t2, obj.γ, 2))
        append!(obj.loss, ssvm_nl(obj.X, obj.Y, obj.w, C, obj.γ))
    end
end


model = GSSVM(linear, X, Y)
fit(model, 0.1, 100, 0.01)

plot(model.loss)
