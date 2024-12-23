export class BingSearchResult {
  async  SearchWeb(searchText: string)  {
    var url : string  = "https://api.bing.microsoft.com/v7.0/search?count=10&offset=0&responseFilter=Webpages&setLang=ja-jp&q=";
    url = url + searchText;
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Ocp-Apim-Subscription-Key': process.env.AZURE_BINGSEARCH_API_KEY
          }});
      return await response.json();
      //const postsData = await response.json();
    } catch (err) {
      console.error(err);
    }
  
  } 
}
