#include<bits/stdc++.h>
 
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    #ifndef ONLINE_JUDGE
        freopen("input.txt","r",stdin);
        freopen("output.txt","w",stdout); 
    #endif

    int tt = 1;
    cin >> tt;
    for (int test = 1; test <= tt; test++) {
        int n, k;
        cin >> n >> k;

        int a[n];
        map<int, int> count;
        int pos1 = 0, pos2 = 0, val = 0;
        for (int i = 0; i < n; ++i)
        {
            cin >> a[i];
            int cur = a[i];
            set<int> s;
            for (int j = k-1; j >= 0; --j)
            {
              int val = (cur & (1 << j)) << (k-1 - j);
              count[]
            }
        }


        
    }

    return 0;
}