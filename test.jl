### Getting it all together

using HigherOrderDerivatives
using Plots
using StatsBase
using CSV
using LinearAlgebra

## functions
p(x, α) = x + 1/α * log(1 + exp(-α*x))

function gssvm_loss(X::AbstractArray, Y::AbstractArray, w::AbstractArray, C::Real, γ::Number, α = 10)
    return C/2 * sum((p.(1 .- Y.*(X * w .* Y .- γ), α)).^2) + 1/2*(w'*w + γ.^2)
end

function ssvm_loss(X::AbstractArray, Y::AbstractArray, w::AbstractArray, C::Real, γ::Number, α = 100)
    return C/2 * sum((p.(1 .- Y.*(X * w .- γ), α)).^2) + 1/2*(w'*w + γ.^2)
end

abstract type SVM end

mutable struct GSSVM <: SVM
    kernel::Function
    X::AbstractArray
    Y::AbstractArray
    w::AbstractArray
    γ::Real
    loss::AbstractArray
    C::Real
    X_train::AbstractArray
    function GSSVM(kernel::Function, X::AbstractArray, Y::AbstractArray)
        X_train = X
        X = kernel(X, X)
        w = rand(size(X,1))
        γ = 1
        loss = []
        C = 0.1
        new(kernel, X, Y, w, γ, loss, C, X_train)
    end
end

mutable struct SSVM <: SVM
    X::AbstractArray
    Y::AbstractArray
    w::AbstractArray
    γ::Real
    loss::AbstractArray
    C::Real
    function SSVM(X::AbstractArray, Y::AbstractArray)
        w = rand(size(X,2))
        γ = 1
        loss = []
        C = 0.1
        new(X, Y, w, γ, loss, C)
    end
end

### kernels


function rbf(X::AbstractArray, XX::AbstractArray; gamma = 0.1)
    kern = [norm(X[i,:] - XX[j,:])^2 for i in 1:size(X,1), j in 1:size(XX,1)]
    return exp.(-(1/(2*gamma)).*kern)
end

function linear(X::AbstractArray, XX::AbstractArray)
    return X'*XX
end

####


function fit!(obj::GSSVM, C::Real)
    obj.C = C
    optimize!(obj, gssvm_loss)
end

function fit!(obj::SSVM, C::Real)
    obj.C = C
    optimize!(obj, ssvm_loss)
end

function optimize!(obj::SVM, loss::Function)
    init = loss(obj.X, obj.Y, obj.w, obj.C, obj.γ)
    append!(obj.loss,init+1)
    append!(obj.loss,init)

    while obj.loss[end-1] - obj.loss[end] > 0.0001
        t1 = w -> loss(obj.X, obj.Y, w, obj.C, obj.γ)
        t2 = γ -> loss(obj.X, obj.Y, obj.w, obj.C, γ)
        d1, d2 = armijo(obj, t1, t2)
        obj.w = obj.w - d1
        obj.γ = obj.γ - d2
        append!(obj.loss, loss(obj.X, obj.Y, obj.w, obj.C, obj.γ))
    end
end

function armijo(obj::SVM, t1::Function, t2::Function)
    hess = hessian(t1, obj.w) ## trying to stabilize hessian
    λ1 = (hess + I(length(obj.w))*norm(hess)*0.1)\ gradient(t1, obj.w)
    λ2 = (derivative(t2, obj.γ) / derivative(t2, obj.γ, 2))
    d1 = linesearch(obj, t1, λ1) * λ1
    d2 = linesearch(obj, t2, λ2) * λ2
    return d1, d2
end

function linesearch(obj::SVM, f::Function, λ, alph = 0.33)
    t = 1
    if λ isa Real
        w = obj.γ
        p = norm(λ)
    else
        w = obj.w
        p = norm(λ)
    end
    while f(w - t.*λ) > f(w) - alph*t*p^2
        t *= 0.5
        print("stepsize = ", t, "\n")
    end
    return t
end


#####

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


X = rand(100, 5)
Y = rand(range(-1, step = 2, 1), 100)
w = rand(size(X,2))

new = SSVM(X_train, Y_train)

fit!(new, 0.01)

plot(new.loss)

function predict(obj::SSVM, X_test)
    pred = X_test*obj.w .- obj.γ
    pred = [if i > 0 (1.) else (-1.) end for i in pred]
    return pred
end

function predict(obj::GSSVM, X_test)
    pred = obj.kernel(X_test, obj.X_train) * (obj.Y .* obj.w) .- obj.γ
    pred = [if i > 0 (1.) else (-1.) end for i in pred]
    return pred
end

function accuracy(pred, Y_test)
    # return 100 - sum(pred - Y_test)/length(pred)
    return 1 - sum(abs.((Y_test + (-1 .*prediction))./2)) / length(prediction)
end

prediction = predict(new, X_test)

accuracy(prediction, Y_test)
