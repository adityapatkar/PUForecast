function [r] = puf_query(c,w)
	[~,n] = size(c);
	phi = ones(1,n+1);
	phi(n+1) = 1;
	for i = n:-1:1
		phi(i) = (2*c(i)-1)*phi(i+1);
	end
	
	r = (phi*w > 0);
end