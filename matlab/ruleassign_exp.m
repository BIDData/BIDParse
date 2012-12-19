function fassign = ruleassign_exp(udarr, lens, pp1, nbs, pp2)
  size1 = size(udarr,1);
  fassign = zeros(size1,6);
  fassign(:,1:3) = ruleblkassign_exp(udarr, lens, pp1);
  indx = fassign(:,1) + lens(1)*(fassign(:,2)-1+lens(2)*(fassign(:,3)-1));
  for i = 1:prod(lens)
    icurr = find(indx == i);
    matb = udarr(icurr, :);
    fassign(icurr, 4:6) = ruleblkassign_exp(matb, nbs, pp2);
  end