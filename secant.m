function res = secant(f, x0, x1, tol)
    res = [x0 x1];
    while abs(x0 - x1) > tol
        f0 = f(x0);
        f1 = f(x1);
        x2 = (x0*f1 - x1*f0)/(f1 - f0);
        x0 = x1;
        x1 = x2;
        res = [res x2];
    end
end
