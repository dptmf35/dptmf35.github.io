source "https://rubygems.org"

# Minimal Mistakes 테마 (remote_theme 사용 시에도 로컬 빌드용으로 명시)
gem "jekyll", "~> 4.3"
gem "minimal-mistakes-jekyll"

# GitHub Pages / 빌드 플러그인
group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-sitemap"
  gem "jekyll-paginate"
  gem "jekyll-include-cache"
  gem "jekyll-gist"
end

# Windows / JRuby 호환용
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]
gem "webrick", "~> 1.8"
