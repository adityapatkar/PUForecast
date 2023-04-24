function j = cost(w, phi, y)
    cur = (phi*w > 0);
    hc = cur - y;
    m = length(y);
    j = (hc' * hc) / (2 * m);
end