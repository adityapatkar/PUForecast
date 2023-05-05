function [w, costs] = gradDesc(rep, lr, w, phi, y, m)
    costs = zeros(rep, 1);
    for r = 1:rep
        cur = (phi*w > 0);
        hc = cur - y;
        temp = sum(hc .* phi);
        w = w - (lr*(2/m)) * temp';
        costs(r) = cost(w,phi,y);
    end
end