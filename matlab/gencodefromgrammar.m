% production version
nnlens=[6 2 2]; % best actual
ntlens=[5 2 2]; % best actual 
tnlens=[6 2 2]; % best actual
ttlens=[4 3 3]; % best actual

nbs=[2 2 2];
ppp=[1 2 3];
stride = 8192;

% nnlens=[2 1 1];

donn = 1;
dont = 1;
dotn = 1;
dott = 1;
spillthresh = 55;

gen = 1;

%codeprefix = 'c:\code\CUDA\C\src\testSparse\';
codeprefix = 'c:\code\Parser\jni\src\rules\';
dataprefix = 'c:\data\Grammar\';

load([dataprefix 'unirules.mat']);

upmap = groot_map;
upcnt = groot_cnt;
ulmap = left_map;
ulcnt = left_cnt;
uleft = left;
upar = groot;
uval = dval; 

load([dataprefix 'binrules.mat']);

roots = {'ROOT_0','EX_0','EX_1'}';


[allmap ip dmy il ir ipu ilu] = cellstrunion(groot_map, roots, left_map, right_map, upmap, ulmap);
% [allmap ip il ir] = cellstrunion(groot_map, left_map, right_map);
allcnt = accumarray([ip ; il ; ir], [groot_cnt; left_cnt; right_cnt]);
darr = [ip(groot) il(left) ir(right)];
uleft = ilu(uleft);
upar = ipu(upar);
nt0 = size(allmap,1);
nt = max(darr(:,1))+length(roots);
ntt = nt0 - nt;
nnsymbols = nt;
ntsymbols = ntt;
uarr = accumarray([upar uleft], uval, [nt0, nt0]);
umodel1 = inv(2*diag(ones(nt0,1),0) - uarr);
umodel = uarr;
xx = uarr-eye(size(uarr,1));
xx = xx * xx; umodel = umodel + umodel * xx;
xx = xx * xx; umodel = umodel + umodel * xx;
xx = xx * xx; umodel = umodel + umodel * xx;
xx = xx * xx; umodel = umodel + umodel * xx;

unn = umodel(1:nt,1:nt);
unt = umodel(1:nt,(nt+1):end);
utt = umodel((nt+1):end,(nt+1):end);
save([dataprefix 'uni_closure.mat'], 'umodel', 'unn', 'unt', 'utt', '-v7.3');

fid = fopen([dataprefix 'fulldict.txt'], 'wt');
for i = 1:length(allmap)
  fprintf(fid,'%s\n', allmap{i});
end
fclose(fid);

if (donn==1)
  lens = nnlens;
  fprintf('\nProcessing NN rules, lens=(%d,%d,%d)\n',lens);
  ii = find(darr(:,2)<=nt & darr(:,3)<=nt);
  udarr = darr(ii,:);
  udval = dval(ii,:);
  nts = [nt nt nt];
  fassign = ruleassign_exp(udarr, lens, ppp, nbs, ppp);
  evalassign;
  codetype='nn';
  if (gen == 1) gencodefromgrammar2; evalassign2; end
end


if (dont==1)
  lens = ntlens;
  fprintf('\nProcessing NT rules, lens=(%d,%d,%d)\n',lens);
  ii = find(darr(:,2)<=nt & darr(:,3)>nt);
  udarr = [darr(ii,1), darr(ii,2), darr(ii,3)-nt];
  udval = dval(ii,:);
  nts = [nt nt ntt];
  fassign = ruleassign_exp(udarr, lens, ppp, nbs, ppp);
  evalassign;
  codetype='nt';
  if (gen == 1) gencodefromgrammar2; evalassign2; end
end

if (dotn==1)
  lens = tnlens;
  fprintf('\nProcessing TN rules, lens=(%d,%d,%d)\n',lens);
  ii = find(darr(:,2)>nt & darr(:,3)<=nt);
  udarr = [darr(ii,1), darr(ii,2)-nt, darr(ii,3)];
  udval = dval(ii,:);
  nts = [nt ntt nt];
  fassign = ruleassign_exp(udarr, lens, ppp, nbs, ppp);
  evalassign;
  codetype='tn';
  if (gen == 1) gencodefromgrammar2; evalassign2; end
end

if (dott==1)
  lens = ttlens;
  fprintf('\nProcessing TT rules, lens=(%d,%d,%d)\n',lens);
  ii = find(darr(:,2)>nt & darr(:,3)>nt);
  udarr = [darr(ii,1), darr(ii,2)-nt, darr(ii,3)-nt];
  udval = dval(ii,:);
  nts = [nt ntt ntt];
  fassign = ruleassign_exp(udarr, lens, ppp, nbs, ppp);
  evalassign;
  codetype='tt';
  if (gen == 1) gencodefromgrammar2; evalassign2; end
end
