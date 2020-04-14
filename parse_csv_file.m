T = readtable("/Users/limingsun/Downloads/Chart-20200129-142502.csv");
T = rmmissing(T);
bb = datetime(T.Time, 'InputFormat', 'hh:mm a', 'Format', 'HH:mm');
bb.Format = 'hh:mm:ss';
T.Time = bb;
T.Time = cellstr(T.Time);
T.Date = cellstr(T.Date);
T.Date = strcat(T.Date, {' '}, T.Time);
T = removevars(T,{'Time'});
T.Date = datenum(T.Date);
T.Properties.VariableNames([1]) = {'Date_Time'};

