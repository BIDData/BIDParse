indx1 = fassign(:,3) + lens(3)*(fassign(:,2)-1 + lens(2)*(fassign(:,1)-1));
indx2 = fassign(:,4) + nbs(1)*(fassign(:,5)-1 + nbs(2)*(fassign(:,6)-1));
gmat = accumarray([indx1 indx2 udarr(:,1)], 1, [prod(lens) prod(nbs) max(udarr(:,1))]);
lmat = accumarray([indx1 indx2 udarr(:,2)], 1, [prod(lens) prod(nbs) max(udarr(:,2))]);
rmat = accumarray([indx1 indx2 udarr(:,3)], 1, [prod(lens) prod(nbs) max(udarr(:,3))]);
gcounts = sum(gmat>0, 3);
lcounts = sum(lmat>0, 3);
rcounts = sum(rmat>0, 3);
rbal = sum(gmat, 3);

plpairs = sparse(udarr(:,1)+nt*(udarr(:,2)-1), indx1+prod(lens)*(indx2-1), ones(size(udarr,1),1), nt*ntt, prod(lens)*prod(nbs));
lrpairs = sparse(udarr(:,2)+ntt*(udarr(:,3)-1), indx1+prod(lens)*(indx2-1), ones(size(udarr,1),1), ntt*ntt, prod(lens)*prod(nbs));
prpairs = sparse(udarr(:,1)+nt*(udarr(:,3)-1), indx1+prod(lens)*(indx2-1), ones(size(udarr,1),1), nt*ntt, prod(lens)*prod(nbs));
pl1 = full(sum(plpairs));
pl2 = full(sum(plpairs>0));
lr1 = full(sum(lrpairs));
lr2 = full(sum(lrpairs>0));
pr1 = full(sum(prpairs));
pr2 = full(sum(prpairs>0));
codesize = 8*(300 + lr1 + lr2 + 2*(gcounts(:) + lcounts(:) + rcounts(:))');
fprintf(['Mean gcnts=%5.3f, lcnts=%5.3f, rcnts=%5.3f\nRMS  gcnts=%5.3f, lcnts=%5.3f, rcnts=%5.3f\n',...
         'Min rules/sblk=%d, max=%d, total=%d\nPL common=%4.3f, LR common=%4.3f, PR common=%4.3f\nConstant mem=%2.1fkB, codesize~%2.1fkB\n',...
         'Mean LR pairs=%4.3f, RMS LR pairs=%4.3f\n'],...
         mean(gcounts(:)),mean(lcounts(:)),mean(rcounts(:)),sqrt(mean(gcounts(:).^2)),sqrt(mean(lcounts(:).^2)),sqrt(mean(rcounts(:).^2)),...
         min(rbal(:)), max(rbal(:)), sum(rbal(:)), mean((pl1-pl2)./pl1), mean((lr1-lr2)./lr1), mean((pr1-pr2)./pr1),...
        4*mean(rbal(:))/1000, mean(codesize)/1000,...
         mean(lr2), sqrt(mean(lr2.^2)));
