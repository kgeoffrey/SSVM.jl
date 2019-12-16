### test smooth ssvm
using HigherOrderDerivatives
using Plots


X = rand(100, 2)
Y = rand(range(-1, step = 2, 1), 100)


p(x, α) = x + 1/α * log(1 + exp(-α*x))

function ssvm(X::AbstractArray, Y::AbstractArray, w::AbstractArray, v::Real, γ::Number, α = 100)
    return v/2 * sum((p.(1 .- Y.*(X*w .- γ), α)).^2) + 1/2*(w'*w + γ.^2)
end


w = rand(size(X,2))
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

    return loss
end


test = optimize(X, Y, 0.1, 200, 0.1)

plot(test)
